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


class HistoryLimiter(Middleware):
    """Truncate message history to a maximum number of events."""

    def __init__(self, max_events: int):
        self._max_events = max_events

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if len(messages) > self._max_events:
            messages = messages[-self._max_events :]
        return await next(messages, ctx, tools)
