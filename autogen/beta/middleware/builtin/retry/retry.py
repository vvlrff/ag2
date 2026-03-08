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


class RetryMiddleware(Middleware):
    """Retry LLM calls on transient failures."""

    def __init__(self, max_retries: int = 3, retry_on: tuple[type[Exception], ...] = (Exception,)):
        self._max_retries = max_retries
        self._retry_on = retry_on

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await next(messages, ctx, tools)
            except self._retry_on as e:
                last_error = e
                if attempt == self._max_retries:
                    raise
        raise last_error  # unreachable but satisfies type checker
