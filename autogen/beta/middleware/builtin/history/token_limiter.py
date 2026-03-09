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


class TokenLimiter(Middleware):
    """Truncate message history to fit within a token budget.

    Uses a simple character-based estimate (``chars_per_token`` chars per token)
    unless a custom tokenizer is provided.
    """

    def __init__(self, max_tokens: int, chars_per_token: int = 4):
        self._max_tokens = max_tokens
        self._chars_per_token = chars_per_token

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        trimmed = list(messages)
        from autogen.beta.events import ModelRequest

        if self._estimate_tokens(trimmed) > self._max_tokens:
            first = trimmed[0]
            has_request = isinstance(first, ModelRequest)

            while len(trimmed) > (2 if has_request else 1) and self._estimate_tokens(trimmed) > self._max_tokens:
                trimmed.pop(1 if has_request else 0)

        return await next(trimmed, ctx, tools)

    def _estimate_tokens(self, messages: list[BaseEvent]) -> int:
        total_chars = sum(len(str(m)) for m in messages)
        return total_chars // self._chars_per_token
