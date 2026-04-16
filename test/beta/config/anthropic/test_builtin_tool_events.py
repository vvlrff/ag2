# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import (
    ServerToolUseBlock,
    TextBlock,
    ToolUseBlock,
    WebSearchToolResultBlock,
)
from anthropic.types.web_search_result_block import WebSearchResultBlock

from autogen.beta.config.anthropic.anthropic_client import AnthropicClient
from autogen.beta.events import (
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ToolCallEvent,
)
from autogen.beta.events.tool_events import ToolResult


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.send = AsyncMock()
    return ctx


def _make_client() -> AnthropicClient:
    return AnthropicClient.__new__(AnthropicClient)


def _make_message(content: list, stop_reason: str = "end_turn") -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = stop_reason
    msg.model = "claude-sonnet-4-6"
    msg.usage = MagicMock()
    msg.usage.model_dump.return_value = {"input_tokens": 10, "output_tokens": 20}
    return msg


def _sent_events(ctx: MagicMock) -> list:
    return [c.args[0] for c in ctx.send.call_args_list]


@pytest.mark.asyncio
class TestProcessResponseBuiltinTools:
    async def test_server_tool_use_emits_builtin_call_event(self) -> None:
        client = _make_client()
        ctx = _make_context()

        msg = _make_message([
            ServerToolUseBlock(
                id="stu_1",
                name="web_search",
                input={"query": "bitcoin price"},
                type="server_tool_use",
            ),
        ])

        await client._process_response(msg, ctx)

        builtin_calls = [e for e in _sent_events(ctx) if isinstance(e, BuiltinToolCallEvent)]
        assert builtin_calls == [
            BuiltinToolCallEvent(id="stu_1", name="web_search", arguments='{"query": "bitcoin price"}'),
        ]

    async def test_web_search_result_emits_builtin_result_event(self) -> None:
        client = _make_client()
        ctx = _make_context()

        msg = _make_message([
            WebSearchToolResultBlock(
                tool_use_id="stu_1",
                type="web_search_tool_result",
                content=[
                    WebSearchResultBlock(
                        title="Bitcoin Price",
                        url="https://example.com",
                        encrypted_content="abc",
                        page_age="1h",
                        type="web_search_result",
                    )
                ],
            ),
        ])

        await client._process_response(msg, ctx)

        builtin_results = [e for e in _sent_events(ctx) if isinstance(e, BuiltinToolResultEvent)]
        # Key order matches WebSearchToolResultBlock.model_dump() output
        expected_content = {
            "content": [
                {
                    "encrypted_content": "abc",
                    "page_age": "1h",
                    "title": "Bitcoin Price",
                    "type": "web_search_result",
                    "url": "https://example.com",
                }
            ],
            "tool_use_id": "stu_1",
            "type": "web_search_tool_result",
        }
        assert builtin_results == [
            BuiltinToolResultEvent(
                parent_id="stu_1",
                name="web_search",
                result=ToolResult(content=expected_content),
            ),
        ]

    async def test_mixed_blocks_routes_regular_tool_to_model_response(self) -> None:
        client = _make_client()
        ctx = _make_context()

        msg = _make_message([
            TextBlock(text="Let me search.", type="text"),
            ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "test"}, type="server_tool_use"),
            WebSearchToolResultBlock(
                tool_use_id="stu_1",
                type="web_search_tool_result",
                content=[],
            ),
            ToolUseBlock(id="tc_1", name="my_func", input={"x": 1}, type="tool_use"),
        ])

        result = await client._process_response(msg, ctx)

        sent_types = {type(e).__name__ for e in _sent_events(ctx)}
        assert sent_types >= {"ModelMessage", "BuiltinToolCallEvent", "BuiltinToolResultEvent"}
        assert list(result.tool_calls.calls) == [
            ToolCallEvent(id="tc_1", name="my_func", arguments='{"x": 1}'),
        ]


@pytest.mark.asyncio
class TestEmitBuiltinToolEvents:
    async def test_emits_for_server_tool_use(self) -> None:
        client = _make_client()
        ctx = _make_context()

        blocks = [
            ServerToolUseBlock(id="stu_1", name="web_search", input={"query": "test"}, type="server_tool_use"),
        ]

        await client._emit_builtin_tool_events(blocks, ctx)

        assert _sent_events(ctx) == [
            BuiltinToolCallEvent(id="stu_1", name="web_search", arguments='{"query": "test"}'),
        ]

    async def test_emits_for_result_block(self) -> None:
        client = _make_client()
        ctx = _make_context()

        blocks = [
            WebSearchToolResultBlock(tool_use_id="stu_1", type="web_search_tool_result", content=[]),
        ]

        await client._emit_builtin_tool_events(blocks, ctx)

        # Key order matches WebSearchToolResultBlock.model_dump() output
        expected_content = {
            "content": [],
            "tool_use_id": "stu_1",
            "type": "web_search_tool_result",
        }
        assert _sent_events(ctx) == [
            BuiltinToolResultEvent(
                parent_id="stu_1",
                name="web_search",
                result=ToolResult(content=expected_content),
            ),
        ]

    async def test_ignores_text_and_tool_use_blocks(self) -> None:
        client = _make_client()
        ctx = _make_context()

        blocks = [
            TextBlock(text="Hello", type="text"),
            ToolUseBlock(id="tc_1", name="func", input={}, type="tool_use"),
        ]

        await client._emit_builtin_tool_events(blocks, ctx)

        ctx.send.assert_not_called()
