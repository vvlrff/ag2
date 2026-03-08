# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, overload

from fast_depends import Provider

from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

from .annotations import Context
from .config import LLMClient, ModelConfig
from .events import (
    BaseEvent,
    ClientToolCall,
    HumanInputRequest,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolError,
    ToolNotFoundEvent,
    ToolResult,
    ToolResults,
)
from .events.conditions import ClassInfo, Condition
from .exceptions import ConfigNotProvidedError, ToolNotFoundError
from .history import History
from .hitl import HumanHook, default_hitl_hook, wrap_hitl
from .middleware import Middleware, _build_chain
from .stream import MemoryStream, Stream
from .tools import FunctionParameters, FunctionTool, Tool, tool

if TYPE_CHECKING:
    from .conversable import ConversableAdapter


class Askable(Protocol):
    async def ask(
        self,
        msg: str,
    ) -> "Conversation":
        raise NotImplementedError


class Conversation(Askable):
    def __init__(
        self,
        message: ModelResponse,
        *,
        ctx: "Context",
        client: "LLMClient",
        agent: "Agent",
    ) -> None:
        self.message = message

        self.ctx = ctx
        self.__client = client
        self.__agent = agent

    async def ask(
        self,
        msg: str,
        *,
        config: ModelConfig | None = None,
        prompt: Iterable[str] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        tools: Iterable[Tool] = (),
    ) -> "Conversation":
        initial_event = ModelRequest(content=msg)

        ctx = self.ctx
        if dependencies:
            ctx.dependencies.update(dependencies)
        if variables:
            ctx.variables.update(variables)
        if prompt:
            ctx.prompt = list(prompt)

        client = config.create() if config else self.__client

        return await self.__agent._execute(
            initial_event,
            ctx=ctx,
            client=client,
            additional_tools=tools,
        )

    @property
    def history(self) -> History:
        return self.ctx.stream.history


PromptHook: TypeAlias = Callable[..., str] | Callable[..., Awaitable[str]]
PromptType: TypeAlias = str | PromptHook


class Agent(Askable):
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[Middleware] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ):
        self.name = name
        self.config = config

        self._agent_dependencies = dependencies or {}
        self._agent_variables = variables or {}

        self.dependency_provider = Provider()
        self.tools = [FunctionTool.ensure_tool(t, provider=self.dependency_provider) for t in tools]

        self.__hitl_hook = wrap_hitl(hitl_hook) if hitl_hook else default_hitl_hook
        self._middleware: list[Middleware] = list(middleware)
        self._observers: list[tuple[ClassInfo | Condition, Callable[..., Any]]] = []

        self._system_prompt: list[str] = []
        self._dynamic_prompt: list[Callable[[ModelRequest, Context], Awaitable[str]]] = []

        if isinstance(prompt, str) or callable(prompt):
            prompt = [prompt]

        for p in prompt:
            if isinstance(p, str):
                self._system_prompt.append(p)
            else:
                self._dynamic_prompt.append(_wrap_prompt_hook(p))

    def hitl_hook(self, func: HumanHook) -> HumanHook:
        if self.__hitl_hook is not default_hitl_hook:
            warnings.warn(
                "You already set HITL hook, provided value overrides it",
                category=RuntimeWarning,
                stacklevel=2,
            )

        self.__hitl_hook = wrap_hitl(func)
        return func

    @overload
    def prompt(
        self,
        func: None = None,
    ) -> Callable[[PromptHook], PromptHook]: ...

    @overload
    def prompt(
        self,
        func: PromptHook,
    ) -> PromptHook: ...

    def prompt(
        self,
        func: PromptHook | None = None,
    ) -> PromptHook | Callable[[PromptHook], PromptHook]:
        def wrapper(f: PromptHook) -> PromptHook:
            self._dynamic_prompt.append(f)
            return f

        if func:
            return wrapper(func)
        return wrapper

    @overload
    def tool(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Tool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Callable[[Callable[..., Any]], Tool]: ...

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        schema: FunctionParameters | None = None,
        sync_to_thread: bool = True,
    ) -> Tool | Callable[[Callable[..., Any]], Tool]:
        def make_tool(f: Callable[..., Any]) -> Tool:
            t = tool(
                f,
                name=name,
                description=description,
                schema=schema,
                sync_to_thread=sync_to_thread,
            )
            t = FunctionTool.ensure_tool(t, provider=self.dependency_provider)
            self.tools.append(t)
            return t

        if function:
            return make_tool(function)

        return make_tool

    def on(self, event_type: type | Condition) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a stream observer for an event type (non-interrupting).

        Usage::

            @agent.on(ModelResponse)
            async def log_response(event: ModelResponse, ctx: Context):
                print(f"Response: {event.message}")
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._observers.append((event_type, func))
            return func

        return decorator

    async def ask(
        self,
        msg: str,
        *,
        stream: Stream | None = None,
        config: ModelConfig | None = None,
        prompt: Iterable[str] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
        tools: Iterable[Tool] = (),
    ) -> "Conversation":
        config = config or self.config
        if not config:
            raise ConfigNotProvidedError()
        client = config.create()

        stream = stream or MemoryStream()

        initial_event = ModelRequest(content=msg)

        ctx = Context(
            stream,
            prompt=list(prompt),
            dependencies=self._agent_dependencies | (dependencies or {}),
            variables=self._agent_variables | (variables or {}),
            dependency_provider=self.dependency_provider,
        )

        if not ctx.prompt:
            ctx.prompt.extend(self._system_prompt)

            for dp in self._dynamic_prompt:
                p = await dp(initial_event, ctx)
                ctx.prompt.append(p)

        return await self._execute(
            initial_event,
            ctx=ctx,
            client=client,
            additional_tools=tools,
        )

    async def _execute(
        self,
        event: BaseEvent,
        *,
        ctx: Context,
        client: LLMClient,
        additional_tools: Iterable[Tool] = (),
    ) -> "Conversation":
        all_tools = self.tools + list(additional_tools)

        # --- Base operations (innermost layer) ---

        async def base_llm_call(messages: list[BaseEvent], ctx: Context, tools: list[Tool]) -> ModelResponse:
            async with ctx.stream.get(ModelResponse) as result:
                await client(*messages, ctx=ctx, tools=tools)
                return await result

        async def base_tool_execute(tool_call: ToolCall, ctx: Context) -> ToolResult | ToolError | ClientToolCall:
            if not any(t.name == tool_call.name for t in all_tools):
                err = ToolNotFoundError(tool_call.name)
                tool_error = ToolNotFoundEvent(
                    parent_id=tool_call.id,
                    name=tool_call.name,
                    content=repr(err),
                    error=err,
                )
                await ctx.send(tool_call)
                await ctx.send(tool_error)
                return tool_error

            async with ctx.stream.get(
                (ToolResult.parent_id == tool_call.id) | (ToolError.parent_id == tool_call.id) | ClientToolCall
            ) as result:
                await ctx.send(tool_call)
                return await result

        async def base_turn(request: BaseEvent, ctx: Context) -> ModelResponse:
            await ctx.send(request)

            messages = list(await ctx.stream.history.get_events())
            response = await llm_call(messages, ctx, all_tools)

            while response.tool_calls and not response.response_force:
                results: list[ToolResult | ToolError] = []
                client_calls: list[ClientToolCall] = []

                for call in response.tool_calls.calls:
                    result = await tool_execute(call, ctx)
                    if isinstance(result, ClientToolCall):
                        client_calls.append(result)
                    else:
                        results.append(result)

                if client_calls:
                    from .events import ToolCalls

                    response = ModelResponse(
                        tool_calls=ToolCalls(calls=client_calls),
                        response_force=True,
                    )
                    await ctx.send(response)
                else:
                    await ctx.send(ToolResults(results=results))
                    messages = list(await ctx.stream.history.get_events())
                    response = await llm_call(messages, ctx, all_tools)

            return response

        # --- Build middleware chains ---

        llm_call = _build_chain(self._middleware, "on_llm_call", base_llm_call)
        tool_execute = _build_chain(self._middleware, "on_tool_execute", base_tool_execute)
        turn = _build_chain(self._middleware, "on_turn", base_turn)

        # --- Execute with stream subscribers ---

        with ExitStack() as stack:
            for tool_obj in all_tools:
                tool_obj.register(stack, ctx)

            stack.enter_context(
                ctx.stream.where(HumanInputRequest).sub_scope(self.__hitl_hook),
            )

            for mw in self._middleware:
                mw.register_observers(stack, ctx)

            for condition, handler in self._observers:
                stack.enter_context(
                    ctx.stream.where(condition).sub_scope(handler),
                )

            response = await turn(event, ctx)

            return Conversation(
                response,
                ctx=ctx,
                agent=self,
                client=client,
            )

    def as_conversable(self) -> "ConversableAdapter":
        from .conversable import ConversableAdapter

        return ConversableAdapter(self)


def _wrap_prompt_hook(func: PromptHook) -> Callable[[ModelRequest, Context], Awaitable[str]]:
    call_model = build_model(func)

    async def wrapper(event: ModelRequest, ctx: Context) -> str:
        async with AsyncExitStack() as stack:
            return await call_model.asolve(
                event,
                stack=stack,
                cache_dependencies={},
                dependency_provider=ctx.dependency_provider,
                **{CONTEXT_OPTION_NAME: ctx},
            )

    return wrapper
