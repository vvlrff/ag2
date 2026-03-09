# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse


class MiddlewareFactory(Protocol):
    def __call__(self, event: "BaseEvent", ctx: "Context") -> "BaseMiddleware": ...


class Middleware(MiddlewareFactory):
    """Public class to simplify middleware registration."""

    def __init__(
        self,
        middleware_cls: type["BaseMiddleware"],
        **kwargs: Any,
    ) -> None:
        self._cls = middleware_cls
        self._options = kwargs

    def __call__(
        self,
        event: "BaseEvent",
        ctx: "Context",
    ) -> "BaseMiddleware":
        return self._cls(event, ctx, **self._options)


AgentTurn: TypeAlias = Callable[["BaseEvent", "Context"], Awaitable["ModelResponse"]]


class BaseMiddleware:
    def __init__(
        self,
        event: "BaseEvent",
        ctx: "Context",
    ) -> None:
        self.initial_event = event
        self.ctx = ctx

    async def on_turn(
        self,
        call_next: AgentTurn,
        event: "BaseEvent",
        ctx: "Context",
    ) -> "ModelResponse":
        return await call_next(event, ctx)
