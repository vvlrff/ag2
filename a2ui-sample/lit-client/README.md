# A2UI Lit Client

A minimal frontend that renders [A2UI](https://a2ui.org) surfaces using the `@a2ui/lit` v0.8 web-component renderer, connected to an AG2 backend via the AG-UI (SSE) protocol.

## Prerequisites

- Node.js 18+
- The AG2 backend running on `http://localhost:8008` (see [Backend Setup](#backend-setup))
- The `@a2ui/lit` package available in `../node_modules/@a2ui/lit` (linked from the parent `a2ui-sample` directory)

## Quick Start

```bash
# Install dependencies
npm install

# Start the dev server
npx vite --port 5174
```

Open http://localhost:5174 and type a prompt like "Show me a weather card for NYC".

## Backend Setup

From the `a2ui-sample/ag2_sample` directory:

```bash
# Install Python dependencies (if not already)
pip install fastapi uvicorn python-dotenv

# Install the A2UI Python SDK (from the A2UI repo)
pip install -e /path/to/A2UI/agent_sdks/python

# Ensure .env has your API key (e.g. GOOGLE_GEMINI_API_KEY)

# Run the backend
python a2ui-sample.py
```

The backend starts on `http://127.0.0.1:8008`.

## How It Works

### Architecture

```
User Input
    │
    ▼
┌─────────────────┐     POST /chat (AG-UI SSE)     ┌──────────────────┐
│  Lit Client      │ ──────────────────────────────► │  AG2 Backend     │
│  (this app)      │ ◄────────────────────────────── │  (a2ui-sample.py)│
│                  │     SSE: TextMessageChunk,      │                  │
│  agui-client.ts  │     ActivitySnapshot events     │  ConversableAgent│
│  app.ts          │                                 │  + A2UI Schema   │
│  theme.ts        │                                 │  + Interceptor   │
└─────────────────┘                                  └──────────────────┘
```

### Data Flow

1. **User submits a query** via the input form in `app.ts`.
2. **`agui-client.ts`** sends a POST request to the AG2 backend with AG-UI message format.
3. **Backend** (`a2ui-sample.py`) passes the query to a `ConversableAgent` whose system prompt includes the A2UI schema and component examples. The LLM generates A2UI JSON wrapped in `<a2ui-json>` tags.
4. **`a2ui_event_interceptor`** on the backend extracts the JSON from the tags and emits it as `ActivitySnapshotEvent`s over SSE.
5. **`agui-client.ts`** parses the SSE stream, collecting A2UI operations from `ACTIVITY_SNAPSHOT` events.
6. **`app.ts`** feeds the operations into `v0_8.Data.createSignalA2uiMessageProcessor()`, which builds a reactive signal-based data model.
7. **`<a2ui-surface>`** components from `@a2ui/lit` render the component tree (Card, Column, Row, Text, Image, Button, etc.) with theming.

### Key Files

| File | Purpose |
|------|---------|
| `src/app.ts` | Main `<a2ui-app>` Lit component. Provides the theme via `ContextProvider`, manages the A2UI message processor, renders surfaces, and handles user actions. |
| `src/agui-client.ts` | AG-UI SSE client. Sends requests to the backend, parses SSE events, and extracts A2UI messages from `ACTIVITY_SNAPSHOT` events. |
| `src/theme.ts` | Theme configuration for `@a2ui/lit` components. Defines styles for Card, Text, Button, Row, Column, etc. |
| `index.html` | HTML shell with CSS custom properties (color palette, grid sizes, fonts). |
| `vite.config.ts` | Vite config. Deduplicates `lit` and `@lit/context` to avoid context mismatch issues. |

### Theme Context

The `@a2ui/lit` components use `@lit/context` to receive the theme. The app provides it using `ContextProvider`:

```typescript
#themeProvider = new ContextProvider(this, {
  context: UI.Context.themeContext,
  initialValue: theme,
});
```

**Important:** Use `ContextProvider` explicitly rather than the `@provide` decorator. The decorator can fail due to TC39 vs experimental decorator compilation differences between your app and `@a2ui/lit`'s pre-compiled components. Also ensure `@lit/context` is in Vite's `resolve.dedupe` array.

### A2UI Message Format (v0.8)

The backend sends three types of A2UI operations:

- **`beginRendering`** — Creates a surface with a root component ID
- **`surfaceUpdate`** — Defines the component tree (Card, Text, Button, etc.)
- **`dataModelUpdate`** — Populates data values referenced by components via paths (e.g. `/temperature`)

### Handling User Actions

When a user clicks a button in the rendered UI, the `<a2ui-surface>` emits an `a2uiaction` event. The app extracts the action context and sends it back to the backend as a JSON message, enabling multi-turn interactive UIs.
