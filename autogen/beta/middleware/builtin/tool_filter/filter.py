# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.middleware.base import Middleware

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse
    from autogen.beta.tools import Tool


class ToolFilter(Middleware):
    """Dynamically filter available tools before the LLM sees them."""

    def __init__(self, filter_fn: Callable[[list[Tool], Context], list[Tool]]):
        self._filter_fn = filter_fn

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        filtered = self._filter_fn(tools, ctx)
        return await next(messages, ctx, filtered)
