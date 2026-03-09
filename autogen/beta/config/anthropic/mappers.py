# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from autogen.beta.builtin_tools import BuiltinTool
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.tools import Tool


def builtin_tool_to_anthropic_params(tool: BuiltinTool) -> dict[str, Any] | None:
    """Convert a BuiltinTool to the Anthropic Messages API tool dict.

    Returns ``None`` for tools not supported by Anthropic.
    """
    if tool.kind == "web_search":
        version = getattr(tool, "anthropic_version", "web_search_20260209")
        params: dict[str, Any] = {"type": version, "name": "web_search"}
        if (max_uses := getattr(tool, "max_uses", None)) is not None:
            params["max_uses"] = max_uses
        if allowed := getattr(tool, "allowed_domains", None):
            params["allowed_domains"] = allowed
        if blocked := getattr(tool, "blocked_domains", None):
            params["blocked_domains"] = blocked
        if loc := getattr(tool, "user_location", None):
            params["user_location"] = {"type": "approximate", **loc}
        return params
    if tool.kind == "code_execution":
        version = getattr(tool, "anthropic_version", "code_execution_20250825")
        return {"type": version, "name": "code_execution"}
    return None


def tool_to_api(t: Tool) -> dict[str, Any]:
    return {
        "name": t.schema.function.name,
        "description": t.schema.function.description,
        "input_schema": t.schema.function.parameters,
    }


def convert_messages(
    messages: tuple[BaseEvent, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, ModelRequest):
            result.append({
                "role": "user",
                "content": message.content,
            })
        elif isinstance(message, ModelResponse):
            content: list[dict[str, Any]] = []
            if message.message:
                content.append({"type": "text", "text": message.message.content})
            for call in message.tool_calls.calls:
                content.append({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments),
                })
            if content:
                result.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolResults):
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.parent_id,
                    "content": r.content,
                }
                for r in message.results
            ]
            result.append({"role": "user", "content": tool_results})

    return result
