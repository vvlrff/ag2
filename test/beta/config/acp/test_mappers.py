# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

from autogen.beta.config.acp.events import ACPAvailableCommands, ACPModeChange, ACPPlan
from autogen.beta.config.acp.mappers import (
    content_blocks_to_files,
    content_blocks_to_text,
    map_session_update,
    map_usage,
)
from autogen.beta.events import ModelMessageChunk, ModelReasoning
from autogen.beta.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent

# Payloads mirror acp model_dump() output (snake_case, `session_update` discriminator).


def test_message_chunk():
    ev = map_session_update({
        "session_update": "agent_message_chunk",
        "content": {"type": "text", "text": "hello"},
    })
    assert isinstance(ev, ModelMessageChunk)
    assert ev.content == "hello"


def test_thought_chunk():
    ev = map_session_update({
        "session_update": "agent_thought_chunk",
        "content": {"type": "text", "text": "thinking"},
    })
    assert isinstance(ev, ModelReasoning)
    assert ev.content == "thinking"


def test_tool_call_start():
    ev = map_session_update({
        "session_update": "tool_call",
        "tool_call_id": "tc1",
        "title": "Edit",
        "raw_input": {"path": "a.py"},
        "status": "pending",
    })
    assert isinstance(ev, BuiltinToolCallEvent)
    assert ev.id == "tc1"
    assert ev.name == "Edit"
    assert ev.serialized_arguments == {"path": "a.py"}


def test_tool_call_progress():
    ev = map_session_update({
        "session_update": "tool_call_update",
        "tool_call_id": "tc1",
        "title": "Edit",
        "status": "completed",
        "content": [{"type": "content", "content": {"type": "text", "text": "done"}}],
    })
    assert isinstance(ev, BuiltinToolResultEvent)
    assert ev.parent_id == "tc1"


def test_plan():
    plan = map_session_update({
        "session_update": "plan",
        "entries": [{"content": "do x", "status": "pending", "priority": "high"}],
    })
    assert isinstance(plan, ACPPlan)
    assert plan.entries[0].content == "do x"
    assert plan.entries[0].priority == "high"


def test_mode_and_commands():
    mode = map_session_update({"session_update": "current_mode_update", "current_mode_id": "edit"})
    assert isinstance(mode, ACPModeChange) and mode.mode_id == "edit"

    cmds = map_session_update({
        "session_update": "available_commands_update",
        "available_commands": [{"name": "/test", "description": "run"}],
    })
    assert isinstance(cmds, ACPAvailableCommands) and cmds.commands == ["/test"]


def test_unknown_update_returns_none():
    assert map_session_update({"session_update": "session_info_update", "title": "x"}) is None
    assert map_session_update({"session_update": "brand_new"}) is None


def test_content_blocks_to_text_concatenates():
    text = content_blocks_to_text([
        {"type": "text", "text": "a"},
        {"type": "image", "data": "x", "mimeType": "image/png"},
        {"type": "text", "text": "b"},
    ])
    assert text == "ab"


def test_content_blocks_to_files_decodes_image():
    data = base64.b64encode(b"img").decode()
    files = content_blocks_to_files([{"type": "image", "data": data, "mimeType": "image/png"}])
    assert files[0].data == b"img"
    assert files[0].metadata["mimeType"] == "image/png"


def test_map_usage():
    usage = map_usage({
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "cached_read_tokens": 2,
        "thought_tokens": 3,
    })
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.cache_read_input_tokens == 2
    assert usage.thinking_tokens == 3


def test_map_usage_none():
    assert not map_usage(None)  # falsy empty Usage
