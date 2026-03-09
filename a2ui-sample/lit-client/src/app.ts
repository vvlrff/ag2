import { SignalWatcher } from "@lit-labs/signals";
import { ContextProvider } from "@lit/context";
import { LitElement, html, css, nothing, unsafeCSS } from "lit";
import { customElement, state } from "lit/decorators.js";
import { repeat } from "lit/directives/repeat.js";
import { v0_8 } from "@a2ui/lit";
import * as UI from "@a2ui/lit/ui";
import { sendMessage, type AGUIResponse } from "./agui-client.js";
import { theme } from "./theme.js";

const BACKEND_URL = "http://localhost:8008/chat";

@customElement("a2ui-app")
export class A2UIApp extends SignalWatcher(LitElement) {
  #themeProvider = new ContextProvider(this, {
    context: UI.Context.themeContext,
    initialValue: theme,
  });

  @state() accessor _loading = false;
  @state() accessor _error: string | null = null;
  @state() accessor _textResponse: string = "";
  @state() accessor _hasContent = false;

  #processor = v0_8.Data.createSignalA2uiMessageProcessor();

  static styles = [
    unsafeCSS(v0_8.Styles.structuralStyles),
    css`
      :host {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        max-width: 720px;
        margin: 0 auto;
        padding: var(--bb-grid-size-4);
        font-family: var(--font-family);
      }

      header {
        text-align: center;
        padding: var(--bb-grid-size-6) 0;
      }
      header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: light-dark(var(--p-40), var(--p-80));
      }
      header p {
        color: light-dark(var(--n-50), var(--n-70));
        font-size: 0.875rem;
        margin-top: var(--bb-grid-size);
      }

      form {
        display: flex;
        gap: var(--bb-grid-size-2);
        padding: var(--bb-grid-size-2) 0;
      }
      input {
        flex: 1;
        padding: var(--bb-grid-size-3) var(--bb-grid-size-4);
        border-radius: 24px;
        border: 1px solid light-dark(var(--n-80), var(--n-30));
        background: light-dark(var(--n-100), var(--n-12));
        color: inherit;
        font-size: 1rem;
        font-family: inherit;
        outline: none;
      }
      input:focus { border-color: var(--p-60); }
      button[type="submit"] {
        padding: var(--bb-grid-size-2) var(--bb-grid-size-5);
        border-radius: 24px;
        border: none;
        background: var(--p-40);
        color: var(--p-100);
        font-size: 0.875rem;
        font-weight: 500;
        cursor: pointer;
        font-family: inherit;
      }
      button[type="submit"]:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }

      .text-response {
        padding: var(--bb-grid-size-3) 0;
        line-height: 1.5;
      }

      #surfaces {
        padding: var(--bb-grid-size-3) 0;
        animation: fadeIn 0.5s ease-out;
      }

      .spinner {
        display: flex;
        justify-content: center;
        padding: var(--bb-grid-size-8) 0;
      }
      .spinner::after {
        content: "";
        width: 32px;
        height: 32px;
        border: 3px solid light-dark(var(--n-90), var(--n-30));
        border-top-color: var(--p-50);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }

      .error {
        background: light-dark(var(--e-90), var(--e-40));
        padding: var(--bb-grid-size-3);
        border-radius: 8px;
        margin: var(--bb-grid-size-2) 0;
      }

      @keyframes spin { to { transform: rotate(360deg); } }
      @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } }
    `,
  ];

  render() {
    return html`
      <header>
        <h1>AG2 + A2UI</h1>
        <p>Powered by the Lit renderer</p>
      </header>

      <form @submit=${this.#onSubmit}>
        <input
          type="text"
          name="query"
          placeholder="Ask for a UI... e.g. 'Show me a weather card for NYC'"
          ?disabled=${this._loading}
          autocomplete="off"
        />
        <button type="submit" ?disabled=${this._loading}>
          ${this._loading ? "..." : "Send"}
        </button>
      </form>

      ${this._error ? html`<div class="error">${this._error}</div>` : nothing}
      ${this._loading ? html`<div class="spinner"></div>` : nothing}
      ${this._textResponse
        ? html`<div class="text-response">${this._textResponse}</div>`
        : nothing}
      ${this.#renderSurfaces()}
    `;
  }

  #renderSurfaces() {
    const surfaces = this.#processor.getSurfaces();
    if (surfaces.size === 0) return nothing;

    return html`<section id="surfaces">
      ${repeat(
        surfaces,
        ([surfaceId]) => surfaceId,
        ([surfaceId, surface]) => {
          return html`<a2ui-surface
            @a2uiaction=${(evt: v0_8.Events.StateEvent<"a2ui.action">) =>
              this.#handleAction(evt, surfaceId)}
            .surfaceId=${surfaceId}
            .surface=${surface}
            .processor=${this.#processor}
          ></a2ui-surface>`;
        },
      )}
    </section>`;
  }

  async #handleAction(
    evt: v0_8.Events.StateEvent<"a2ui.action">,
    surfaceId: string,
  ) {
    const context: Record<string, unknown> = {};
    if (evt.detail.action.context) {
      for (const item of evt.detail.action.context) {
        if (item.value.literalString) {
          context[item.key] = item.value.literalString;
        } else if (item.value.literalNumber) {
          context[item.key] = item.value.literalNumber;
        } else if (item.value.literalBoolean) {
          context[item.key] = item.value.literalBoolean;
        } else if (item.value.path) {
          const path = this.#processor.resolvePath(
            item.value.path,
            evt.detail.dataContextPath,
          );
          context[item.key] = this.#processor.getData(
            evt.detail.sourceComponent,
            path,
            surfaceId,
          );
        }
      }
    }

    const actionMessage = JSON.stringify({
      userAction: {
        name: evt.detail.action.name,
        surfaceId,
        sourceComponentId: (evt.composedPath()[0] as HTMLElement)?.id,
        timestamp: new Date().toISOString(),
        context,
      },
    });

    await this.#send(actionMessage);
  }

  async #onSubmit(e: Event) {
    e.preventDefault();
    const form = e.target as HTMLFormElement;
    const input = form.querySelector("input") as HTMLInputElement;
    const query = input.value.trim();
    if (!query) return;
    input.value = "";
    await this.#send(query);
  }

  async #send(message: string) {
    this._loading = true;
    this._error = null;
    this._textResponse = "";
    this.#processor.clearSurfaces();

    try {
      const result: AGUIResponse = await sendMessage(BACKEND_URL, message);

      console.log("[a2ui-client] Response:", result);

      if (result.text) {
        this._textResponse = result.text;
      }

      if (result.a2uiMessages.length > 0) {
        console.log("[a2ui-client] Processing", result.a2uiMessages.length, "A2UI messages");
        this.#processor.processMessages(
          result.a2uiMessages as v0_8.Types.ServerToClientMessage[],
        );
      }

      this._hasContent = true;
    } catch (err) {
      this._error = `Error: ${(err as Error).message}`;
      console.error(err);
    } finally {
      this._loading = false;
    }
  }
}
