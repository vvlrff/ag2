# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, Field

from autogen.beta.agent import Agent
from autogen.beta.config.config import ModelConfig
from autogen.beta.middleware.base import MiddlewareFactory
from autogen.beta.response.proto import ResponseProto
from autogen.beta.response.schema import RawSchema
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.tool import Tool

__all__ = ("AgentSpec", "ResponseSchemaSpec")


def _is_json_safe(value: object) -> bool:
    """Quick check: can *value* survive a JSON round-trip?"""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


class ResponseSchemaSpec(BaseModel):
    """JSON-serializable description of a response schema."""

    name: str
    description: str | None = None
    json_schema: dict[str, Any]

    def to_response_schema(self) -> ResponseProto[str]:
        """Reconstruct a ``RawSchema`` from this spec."""

        return RawSchema(
            self.json_schema,
            name=self.name,
            description=self.description,
        )


class AgentSpec(BaseModel):
    """JSON-serializable specification of an Agent.

    Captures the declarative, data-only parts of an ``Agent``: name, prompt,
    tool references (by name), config, response schema, and variables.

    Non-serializable parts (middleware, callbacks, dependencies, dynamic prompts)
    are intentionally excluded and must be supplied at reconstruction time via
    :meth:`to_agent`.
    """

    name: str
    prompt: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    response_schema: ResponseSchemaSpec | None = None
    variables: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent(cls, agent: Agent) -> "AgentSpec":
        """Create an ``AgentSpec`` from a live ``Agent`` instance.

        Only serializable state is captured. Dynamic prompts, middleware,
        callbacks, and dependencies are dropped.
        """

        # Tool names
        tool_names: list[str] = []
        for t in agent.tools:
            if isinstance(t, FunctionTool):
                tool_names.append(t.schema.function.name)

        # Response schema
        rs_spec: ResponseSchemaSpec | None = None
        rs = agent._response_schema
        if rs is not None and rs.json_schema is not None:
            rs_spec = ResponseSchemaSpec(
                name=rs.name,
                description=rs.description,
                json_schema=rs.json_schema,
            )

        # Variables (only JSON-safe values)
        variables: dict[str, Any] = {}
        for k, v in agent._agent_variables.items():
            if isinstance(k, str) and _is_json_safe(v):
                variables[k] = v

        return cls(
            name=agent.name,
            prompt=list(agent._system_prompt),
            tool_names=tool_names,
            response_schema=rs_spec,
            variables=variables,
        )

    def to_agent(
        self,
        *,
        available_tools: Iterable[Tool | Callable[..., Any]] = (),
        config: ModelConfig | None = None,
        middleware: Iterable[MiddlewareFactory] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[str, Any] | None = None,
        response_schema: ResponseProto[Any] | type | None = None,
    ) -> Agent:
        """Reconstruct an ``Agent`` from this spec.

        Parameters
        ----------
        available_tools:
            Pool of tools to match against :attr:`tool_names`. Each tool is
            identified by its function name (``__name__`` for callables,
            ``schema.function.name`` for ``FunctionTool``).
        config:
            Override the config from the spec. If ``None``, the spec's config
            is used (if present).
        middleware:
            Middleware factories to attach (not stored in the spec).
        dependencies:
            Dependency dict to inject (not stored in the spec).
        variables:
            Extra variables merged on top of the spec's variables.
        response_schema:
            Override the response schema from the spec.
        """

        # Build name -> tool index from available_tools
        tool_index: dict[str, Tool | Callable[..., Any]] = {}
        for t in available_tools:
            if isinstance(t, FunctionTool):
                tool_index[t.schema.function.name] = t
            elif callable(t):
                name = getattr(t, "__name__", None)
                if name is not None:
                    tool_index[name] = t

        # Resolve tools by name
        resolved_tools: list[Tool | Callable[..., Any]] = []
        missing: list[str] = []
        for name in self.tool_names:
            if name in tool_index:
                resolved_tools.append(tool_index[name])
            else:
                missing.append(name)

        if missing:
            raise ValueError(f"Could not resolve tool(s): {missing}. Available: {sorted(tool_index)}")

        # Response schema: explicit param > spec > None
        final_rs = response_schema
        if final_rs is None and self.response_schema is not None:
            final_rs = self.response_schema.to_response_schema()

        # Variables: spec + overrides
        final_variables = dict(self.variables)
        if variables:
            final_variables.update(variables)

        return Agent(
            name=self.name,
            prompt=list(self.prompt),
            config=config,
            tools=resolved_tools,
            middleware=middleware,
            dependencies=dependencies,
            variables=final_variables or None,
            response_schema=final_rs,
        )
