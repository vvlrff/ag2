# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.middleware.base import Middleware

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse, ToolCall, ToolError, ToolResult
    from autogen.beta.tools import Tool


class LoggingMiddleware(Middleware):
    """Log LLM calls, tool executions, and turns with timing."""

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger("autogen.beta.middleware")

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._logger.info("LLM call: %d messages, %d tools", len(messages), len(tools))
        t0 = time.perf_counter()
        response = await next(messages, ctx, tools)
        elapsed = time.perf_counter() - t0
        self._logger.info("LLM response in %.2fs", elapsed)
        return response

    async def on_tool_execute(
        self,
        tool_call: ToolCall,
        ctx: Context,
        next: Callable[..., Awaitable[ToolResult | ToolError]],
    ) -> ToolResult | ToolError:
        self._logger.info("Tool execute: %s(%s)", tool_call.name, tool_call.arguments)
        result = await next(tool_call, ctx)
        self._logger.info("Tool result: %s", type(result).__name__)
        return result

    async def on_turn(
        self,
        request: BaseEvent,
        ctx: Context,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._logger.info("Turn started")
        response = await next(request, ctx)
        self._logger.info("Turn finished")
        return response
