# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    ServerToolUseBlock,
    TextBlock,
    TextEditorCodeExecutionToolResultBlock,
    ToolUseBlock,
    WebFetchToolResultBlock,
    WebSearchToolResultBlock,
)
from anthropic.types.bash_code_execution_tool_result_error import BashCodeExecutionToolResultError
from anthropic.types.code_execution_tool_result_error import CodeExecutionToolResultError
from anthropic.types.text_editor_code_execution_tool_result_error import TextEditorCodeExecutionToolResultError
from anthropic.types.web_fetch_tool_result_error_block import WebFetchToolResultErrorBlock

from autogen.beta.config.anthropic import AnthropicClient
from autogen.beta.events import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ToolCallEvent,
    ToolCallsEvent,
)
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.web_fetch import WEB_FETCH_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


def _make_message(content: list[Any]) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = "end_turn"
    msg.model = "claude-sonnet-4-6"
    msg.usage = MagicMock()
    msg.usage.model_dump.return_value = {"input_tokens": 10, "output_tokens": 20}
    return msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "expected_name"),
    [
        pytest.param("web_search", WEB_SEARCH_TOOL_NAME, id="web_search"),
        pytest.param("web_fetch", WEB_FETCH_TOOL_NAME, id="web_fetch"),
        pytest.param("code_execution", CODE_EXECUTION_TOOL_NAME, id="code_execution"),
        pytest.param("bash_code_execution", CODE_EXECUTION_TOOL_NAME, id="bash"),
        pytest.param("text_editor_code_execution", CODE_EXECUTION_TOOL_NAME, id="text_editor"),
    ],
)
async def test_server_tool_use_emits_builtin_call_event(provider_name: str, expected_name: str) -> None:
    ctx = AsyncMock()
    client = AnthropicClient(api_key="test", prompt_caching=False)
    msg = _make_message([
        ServerToolUseBlock(id="stu_1", name=provider_name, input={"q": "x"}, type="server_tool_use"),
    ])

    await client._process_response(msg, ctx)

    [event] = [c.args[0] for c in ctx.send.call_args_list]
    assert event == BuiltinToolCallEvent(id="stu_1", name=expected_name, arguments='{"q": "x"}')
    assert event.provider_data == {"provider_name": provider_name}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block", "expected_name"),
    [
        pytest.param(
            WebSearchToolResultBlock(tool_use_id="stu_1", type="web_search_tool_result", content=[]),
            WEB_SEARCH_TOOL_NAME,
            id="web_search",
        ),
        pytest.param(
            WebFetchToolResultBlock(
                tool_use_id="stu_1",
                type="web_fetch_tool_result",
                content=WebFetchToolResultErrorBlock(
                    error_code="unavailable",
                    type="web_fetch_tool_result_error",
                ),
            ),
            WEB_FETCH_TOOL_NAME,
            id="web_fetch",
        ),
        pytest.param(
            CodeExecutionToolResultBlock(
                tool_use_id="stu_1",
                type="code_execution_tool_result",
                content=CodeExecutionToolResultError(
                    error_code="unavailable",
                    type="code_execution_tool_result_error",
                ),
            ),
            CODE_EXECUTION_TOOL_NAME,
            id="code_execution",
        ),
        pytest.param(
            BashCodeExecutionToolResultBlock(
                tool_use_id="stu_1",
                type="bash_code_execution_tool_result",
                content=BashCodeExecutionToolResultError(
                    error_code="unavailable",
                    type="bash_code_execution_tool_result_error",
                ),
            ),
            CODE_EXECUTION_TOOL_NAME,
            id="bash",
        ),
        pytest.param(
            TextEditorCodeExecutionToolResultBlock(
                tool_use_id="stu_1",
                type="text_editor_code_execution_tool_result",
                content=TextEditorCodeExecutionToolResultError(
                    error_code="file_not_found",
                    type="text_editor_code_execution_tool_result_error",
                ),
            ),
            CODE_EXECUTION_TOOL_NAME,
            id="text_editor",
        ),
    ],
)
async def test_result_block_emits_builtin_result_event(block: Any, expected_name: str) -> None:
    ctx = AsyncMock()
    client = AnthropicClient(api_key="test", prompt_caching=False)
    msg = _make_message([block])

    await client._process_response(msg, ctx)

    assert [c.args[0] for c in ctx.send.call_args_list] == [
        BuiltinToolResultEvent(
            parent_id="stu_1",
            name=expected_name,
            result=ToolResult(content=block.model_dump(exclude_none=True)),
        ),
    ]


@pytest.mark.asyncio
async def test_mixed_blocks_route_correctly() -> None:
    """Regular tool_use goes to result.tool_calls; builtin blocks go to the stream."""
    ctx = AsyncMock()
    client = AnthropicClient(api_key="test", prompt_caching=False)
    msg = _make_message([
        TextBlock(text="Let me search.", type="text"),
        ServerToolUseBlock(id="stu_1", name="web_search", input={"q": "x"}, type="server_tool_use"),
        WebSearchToolResultBlock(tool_use_id="stu_1", type="web_search_tool_result", content=[]),
        ToolUseBlock(id="tc_1", name="my_func", input={"x": 1}, type="tool_use"),
    ])

    result = await client._process_response(msg, ctx)

    assert result.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="tc_1", name="my_func", arguments='{"x": 1}'),
    ])
    assert result.message == ModelMessage("Let me search.")
    sent_types = [type(c.args[0]) for c in ctx.send.call_args_list]
    assert sent_types == [BuiltinToolCallEvent, BuiltinToolResultEvent, ModelMessage]
