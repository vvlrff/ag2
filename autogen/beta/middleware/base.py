# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse, ToolCall, ToolError, ToolResult
    from autogen.beta.tools import Tool


class Middleware:
    """Base class for middleware. Override only the hooks you need.

    Middleware wraps agent operations using an onion model:
    ``[A, B, C]`` → A wraps B wraps C wraps base operation.

    First middleware in list = outermost wrapper (sees everything first/last).
    """

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wraps a single LLM invocation.

        Override to:
        - Modify ``messages`` (history) before passing to next
        - Modify ``tools`` list before passing to next
        - Inspect/transform the ModelResponse after next returns
        - Retry by calling ``next()`` multiple times
        - Block by NOT calling ``next()`` and returning a synthetic ModelResponse
        """
        return await next(messages, ctx, tools)

    async def on_tool_execute(
        self,
        tool_call: ToolCall,
        ctx: Context,
        next: Callable[..., Awaitable[ToolResult | ToolError]],
    ) -> ToolResult | ToolError:
        """Wraps a single tool execution."""
        return await next(tool_call, ctx)

    async def on_turn(
        self,
        request: BaseEvent,
        ctx: Context,
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wraps an entire agent turn (request -> tool loops -> final response).

        Outermost hook — good for timing, circuit breakers, rate limiting.
        """
        return await next(request, ctx)

    def register_observers(self, stack: ExitStack, ctx: Context) -> None:
        """Register stream subscribers for observation (non-interrupting).

        Use this for logging, metrics, streaming chunk observation, etc.
        """


def _build_chain(
    middleware_list: list[Middleware], method_name: str, base_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """Build an onion-model chain: [A, B, C] -> A wraps B wraps C wraps base_fn."""
    fn = base_fn
    default_impl = getattr(Middleware, method_name)

    for mw in reversed(middleware_list):
        if getattr(type(mw), method_name) is default_impl:
            continue
        prev = fn
        bound = getattr(mw, method_name)

        async def chained(*args: Any, _bound: Any = bound, _prev: Any = prev) -> Any:
            return await _bound(*args, next=_prev)

        fn = chained
    return fn
