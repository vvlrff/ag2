# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.middleware.base import Middleware

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse
    from autogen.beta.tools import Tool


class TimeoutMiddleware(Middleware):
    """Wraps LLM calls with an asyncio timeout."""

    def __init__(self, seconds: float):
        self._seconds = seconds

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await asyncio.wait_for(next(messages, ctx, tools), timeout=self._seconds)
