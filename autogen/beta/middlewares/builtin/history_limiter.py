# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from itertools import dropwhile

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, ToolResults
from autogen.beta.middlewares.base import BaseMiddleware, LLMCall, MiddlewareFactory


class HistoryLimiter(MiddlewareFactory):
    def __init__(self, max_events: int) -> None:
        if max_events < 1:
            raise ValueError("max_events must be greater than 0")
        self._max_events = max_events

    def __call__(self, event: "BaseEvent", ctx: "Context") -> "BaseMiddleware":
        return _HistoryLimiter(event, ctx, self._max_events)


class _HistoryLimiter(BaseMiddleware):
    """Truncate message history to a maximum number of events."""

    def __init__(self, event: "BaseEvent", ctx: "Context", max_events: int) -> None:
        super().__init__(event, ctx)
        self._max_events = max_events

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        ctx: Context,
    ) -> ModelResponse:
        if len(events) <= self._max_events:
            return await call_next(events, ctx)

        first = events[0]
        if isinstance(first, ModelRequest):
            trimmed = [first]
            if self._max_events > 1:
                trimmed.extend(
                    dropwhile(
                        lambda x: isinstance(x, ToolResults),
                        events[-(self._max_events - 1) :],
                    )
                )
        else:
            trimmed = list(
                dropwhile(
                    lambda x: isinstance(x, ToolResults),
                    events[-self._max_events :],
                )
            )

        return await call_next(trimmed, ctx)
