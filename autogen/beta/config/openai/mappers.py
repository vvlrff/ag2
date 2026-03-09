# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any

from autogen.beta.builtin_tools import BuiltinTool
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def builtin_tool_to_responses_params(tool: BuiltinTool) -> dict[str, Any] | None:
    """Convert a BuiltinTool to the OpenAI Responses API tool dict.

    Returns ``None`` for tools not supported by the Responses API.
    """
    if tool.kind == "web_search":
        version = getattr(tool, "openai_version", "web_search_preview")
        params: dict[str, Any] = {
            "type": version,
            "search_context_size": getattr(tool, "search_context_size", "medium"),
        }
        if loc := getattr(tool, "user_location", None):
            params["user_location"] = {"type": "approximate", **loc}
        if allowed := getattr(tool, "allowed_domains", None):
            params["allowed_domains"] = allowed
        return params
    if tool.kind == "code_execution":
        container_id = getattr(tool, "container_id", None)
        container = container_id if container_id else {"type": "auto"}
        return {"type": "code_interpreter", "container": container}
    return None


def events_to_responses_input(messages: Sequence[BaseEvent]) -> list[dict[str, Any]]:
    """Convert a sequence of events to Responses API input items."""
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append({
                "role": "user",
                "content": [{"type": "input_text", "text": message.content}],
            })
        elif isinstance(message, ModelResponse):
            # Reconstruct assistant message
            content: list[dict[str, Any]] = []
            if message.message:
                content.append({"type": "output_text", "text": message.message.content})
            if content:
                result.append({
                    "role": "assistant",
                    "content": content,
                })
            # Add function call items from the response
            for call in message.tool_calls.calls:
                result.append({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                })
        elif isinstance(message, ToolResults):
            for r in message.results:
                result.append({
                    "type": "function_call_output",
                    "call_id": r.parent_id,
                    "output": r.content,
                })

    return result


def convert_messages(
    system_prompt: Iterable[str],
    messages: tuple[BaseEvent, ...],
) -> list[dict[str, str]]:
    # legacy prompt message format
    result: list[dict[str, str]] = [{"content": "\n".join(system_prompt), "role": "system"}]

    for message in messages:
        if isinstance(message, (ModelRequest, ModelResponse)):
            result.append(message.to_api())
        elif isinstance(message, ToolResults):
            for r in message.results:
                result.append(r.to_api())

    return result


def tool_to_api(t: Tool) -> dict[str, Any]:
    """Chat Completions API tool format."""
    return {
        "type": "function",
        "function": {
            "name": t.schema.function.name,
            "description": t.schema.function.description,
            "parameters": {"additionalProperties": False} | t.schema.function.parameters,
        },
    }


def tool_to_responses_api(t: Tool) -> dict[str, Any]:
    """Responses API tool format — name/description at top level."""
    return {
        "type": "function",
        "name": t.schema.function.name,
        "description": t.schema.function.description,
        "parameters": {"additionalProperties": False} | t.schema.function.parameters,
    }
