import json
import os
import re

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

from a2ui.core.schema.catalog import CatalogConfig
from a2ui.core.schema.manager import A2uiSchemaManager
from ag_ui.core import ActivitySnapshotEvent
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from autogen import ConversableAgent, LLMConfig
from autogen.ag_ui import AGUIStream, RunAgentInput
from autogen.agentchat.remote import ServiceResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BASE = Path(__file__).resolve().parent
_A2UI_SPEC = _BASE / "specification" / "v0_8" / "json"


def add_catalog_id(schema):
    if schema is not None and "components" in schema and "catalogId" not in schema:
        schema["catalogId"] = "basic"
    return schema


schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[
        CatalogConfig.from_path(
            name="basic",
            catalog_path=str(_A2UI_SPEC / "standard_catalog_definition.json"),
            examples_path=str(_A2UI_SPEC / "catalogs" / "basic" / "examples"),
        )
    ],
    schema_modifiers=[add_catalog_id],
)

instruction = schema_manager.generate_system_prompt(
    role_description="You are a helpful AI assistant that generates rich UIs.",
    workflow_description="Analyze the user request and return an A2UI JSON payload to render interactive interfaces when appropriate.",
    ui_description="Use the provided A2UI components to structure the output.",
    include_schema=True,  # Injects the raw JSON schema
    include_examples=True,  # Injects few-shot examples
)

print(instruction)


agent = ConversableAgent(
    name="a2ui_support_bot",
    system_message=instruction,
    llm_config=LLMConfig({
        "api_type": "google",
        "model": "gemini-3.1-flash-lite-preview",
        "api_key": os.environ.get("GOOGLE_GEMINI_API_KEY"),
    }),
)


_A2UI_TAG_RE = re.compile(r"<a2ui-json>\s*([\s\S]*?)\s*</a2ui-json>")


async def a2ui_event_interceptor(response: ServiceResponse) -> AsyncIterator[ActivitySnapshotEvent]:
    if response.message and (data := response.message.get("content")):
        matches = _A2UI_TAG_RE.findall(data)
        if matches:
            for raw_json in matches:
                try:
                    a2ui_data = json.loads(raw_json)
                    print(f"[a2ui] Emitting ActivitySnapshot with {len(a2ui_data)} operations")
                    yield ActivitySnapshotEvent(
                        message_id=str(uuid4()),
                        activity_type="a2ui-surface",
                        content={"operations": a2ui_data},
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[a2ui] Failed to parse JSON block: {e}")
            # Keep only the text outside the a2ui-json tags
            remaining_text = _A2UI_TAG_RE.sub("", data).strip()
            if remaining_text:
                response.message["content"] = remaining_text
            else:
                response.message = None
        else:
            print(f"[a2ui] No <a2ui-json> tags found in response ({len(data)} chars)")


# forwarded_props = {
#     "a2uiAction": {
#         "userAction": {
#             "name": "see_examples",
#             "sourceComponentId": "examples-btn",
#             "surfaceId": "greeting-surface",
#             "timestamp": "2026-03-05T17:53:14.947Z",
#             "context": {},
#         }
#     }
# }

# forwarded_props = {
#     "a2uiAction": {
#         "userAction": {
#             "name": "submit_query",
#             "sourceComponentId": "send-btn",
#             "surfaceId": "assistant-welcome",
#             "timestamp": "2026-03-05T17:57:21.998Z",
#             "context": {"query": "dasdasda"},
#         }
#     }
# }

stream = AGUIStream(
    agent,
    event_interceptors=[a2ui_event_interceptor],
)


@app.post("/chat")
async def run_agent(
    message: RunAgentInput,
    accept: str | None = Header(None),
) -> StreamingResponse:
    return StreamingResponse(
        stream.dispatch(message, accept=accept),
        media_type=accept or "text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)
