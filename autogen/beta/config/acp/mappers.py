# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Pure translation between ACP payloads and AG2 beta events.

Functions here take plain dicts (the SDK's ``model.model_dump()`` output, which
uses snake_case field names and a ``session_update`` discriminator) so they are
unit-testable without importing ``acp`` or spawning a subprocess.
"""

import base64
import json

from autogen.beta.events import BaseEvent, ModelMessageChunk, ModelReasoning
from autogen.beta.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent, ToolResult
from autogen.beta.events.types import BinaryResult, Usage

from .events import ACPAvailableCommands, ACPModeChange, ACPPlan, ACPPlanEntry


def content_blocks_to_text(blocks: list[dict] | None) -> str:
    """Concatenate the text of any ``text`` content blocks; ignore the rest."""
    return "".join(b.get("text", "") for b in (blocks or ()) if b.get("type") == "text")


def content_blocks_to_files(blocks: list[dict] | None) -> list[BinaryResult]:
    """Decode ``image``/``audio`` content blocks into binary results."""
    files: list[BinaryResult] = []
    for b in blocks or ():
        if b.get("type") in ("image", "audio") and b.get("data") is not None:
            files.append(
                BinaryResult(
                    data=base64.b64decode(b["data"]),
                    metadata={"mimeType": b.get("mimeType") or b.get("mime_type", "")},
                )
            )
    return files


def _tool_content_text(content: list[dict] | None) -> str:
    """Extract text from a tool call's ``content`` list (ContentToolCallContent)."""
    out: list[str] = []
    for item in content or ():
        inner = item.get("content")
        if isinstance(inner, dict) and inner.get("type") == "text":
            out.append(inner.get("text", ""))
    return "".join(out)


def map_usage(usage: dict | None) -> Usage:
    """Map an ACP ``Usage`` dict onto AG2's :class:`Usage` (absent -> empty)."""
    if not usage:
        return Usage()
    return Usage(
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        cache_read_input_tokens=usage.get("cached_read_tokens"),
        cache_creation_input_tokens=usage.get("cached_write_tokens"),
        thinking_tokens=usage.get("thought_tokens"),
    )


def map_session_update(update: dict) -> BaseEvent | None:
    """Translate one ACP ``session/update`` payload into an AG2 event.

    Returns ``None`` for variants with no meaningful AG2 representation
    (``user_message_chunk``, ``usage_update``, ``session_info_update``, and any
    future/unknown variant). ``usage_update`` is handled out-of-band by the
    client via :func:`map_usage`.
    """
    kind = update.get("session_update")

    if kind == "agent_message_chunk":
        return ModelMessageChunk(content_blocks_to_text([update["content"]]))

    if kind == "agent_thought_chunk":
        return ModelReasoning(content_blocks_to_text([update["content"]]))

    if kind == "tool_call":
        return BuiltinToolCallEvent(
            id=update.get("tool_call_id", ""),
            name=update.get("title") or "tool",
            arguments=json.dumps(update.get("raw_input") or {}),
        )

    if kind == "tool_call_update":
        text = _tool_content_text(update.get("content")) or (update.get("status") or "")
        return BuiltinToolResultEvent(
            parent_id=update.get("tool_call_id", ""),
            name=update.get("title"),
            result=ToolResult(text),
        )

    if kind == "plan":
        return ACPPlan(
            entries=[
                ACPPlanEntry(
                    content=e.get("content", ""),
                    status=e.get("status", ""),
                    priority=e.get("priority"),
                )
                for e in update.get("entries", [])
            ]
        )

    if kind == "current_mode_update":
        return ACPModeChange(mode_id=update.get("current_mode_id", ""))

    if kind == "available_commands_update":
        return ACPAvailableCommands(commands=[c.get("name", "") for c in update.get("available_commands", [])])

    return None
