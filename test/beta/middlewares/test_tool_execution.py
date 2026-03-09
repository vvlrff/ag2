# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, Context
from autogen.beta.events import BaseEvent, ToolCall, ToolResult
from autogen.beta.middlewares import AgentTurn, BaseMiddleware, Middleware
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools import tool


@pytest.mark.asyncio()
async def test_tool_execution_middleware(mock: MagicMock) -> None:
    class MockMiddleware(BaseMiddleware):
        def __init__(
            self,
            event: BaseEvent,
            ctx: Context,
            mock: MagicMock,
        ) -> None:
            super().__init__(event, ctx)
            self.mock = mock

        async def on_tool_execution(
            self,
            call_next: AgentTurn,
            event: ToolCall,
            ctx: Context,
        ) -> ToolResult:
            self.mock.enter(event.name)
            r = await call_next(event, ctx)
            self.mock.exit(r.content)
            return r

    def my_tool() -> str:
        return "tool executed"

    agent = Agent(
        "",
        config=TestConfig(
            ToolCall(name="my_tool"),
            "result",
        ),
        tools=[my_tool],
        middlewares=[Middleware(MockMiddleware, mock=mock)],
    )

    await agent.ask("Hi!")

    mock.enter.assert_called_once_with("my_tool")
    mock.exit.assert_called_once_with('"tool executed"')


@pytest.mark.asyncio()
async def test_capture_tool_execution_error(mock: MagicMock) -> None:
    class MockMiddleware(BaseMiddleware):
        def __init__(
            self,
            event: BaseEvent,
            ctx: Context,
            mock: MagicMock,
        ) -> None:
            super().__init__(event, ctx)
            self.mock = mock

        async def on_tool_execution(
            self,
            call_next: AgentTurn,
            event: ToolCall,
            ctx: Context,
        ) -> ToolResult:
            r = await call_next(event, ctx)
            self.mock.exit(repr(r.error))
            # suppress the error
            return ToolResult(parent_id=event.id, name=event.name, raw_content="tool executed")

    def my_tool() -> str:
        raise ValueError("tool execution error")

    tracking_config = TrackingConfig(TestConfig(ToolCall(name="my_tool"), "result"))

    agent = Agent(
        "",
        config=tracking_config,
        tools=[my_tool],
        middlewares=[Middleware(MockMiddleware, mock=mock)],
    )

    await agent.ask("Hi!")

    mock.exit.assert_called_once_with("ValueError('tool execution error')")

    tool_results_event = tracking_config.mock.call_args_list[1].args[0]
    assert tool_results_event.results[0].content == '"tool executed"'


@pytest.mark.asyncio()
async def test_tool_execution_middleware_mutates_arguments_and_result() -> None:
    class MutatingToolMiddleware(BaseMiddleware):
        async def on_tool_execution(
            self,
            call_next: AgentTurn,
            event: ToolCall,
            ctx: Context,
        ) -> ToolResult:
            event.serialized_arguments["x"] += 1
            result = await call_next(event, ctx)
            result.raw_content += "!"
            return result

    recorded_args = MagicMock()

    @tool
    def my_tool(x: int) -> str:
        recorded_args(x)
        return f"{x}"

    tracking_config = TrackingConfig(
        TestConfig(
            ToolCall(name="my_tool", arguments='{"x": 1}'),
            "done",
        ),
    )

    agent = Agent(
        "",
        config=tracking_config,
        tools=[my_tool],
        middlewares=[MutatingToolMiddleware, MutatingToolMiddleware, MutatingToolMiddleware],
    )

    await agent.ask("Hi!")

    # Argument mutation: x starts as 1 and is incremented by each middleware.
    recorded_args.assert_called_once_with(4)

    # Result mutation: ToolResults passed back to the client should reflect
    # the mutated ToolResult content.
    tool_results_event = tracking_config.mock.call_args_list[1].args[0]
    assert tool_results_event.results[0].content == '"4!!!"'
