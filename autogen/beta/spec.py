# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from autogen.beta.agent import Agent
    from autogen.beta.config.config import ModelConfig
    from autogen.beta.middleware.base import MiddlewareFactory
    from autogen.beta.response.proto import ResponseProto
    from autogen.beta.tools.tool import Tool

__all__ = ("AgentSpec", "ConfigSpec", "ResponseSchemaSpec")


_CONFIG_CLASS_TO_PROVIDER: dict[str, str] = {
    "OpenAIConfig": "openai",
    "OpenAIResponsesConfig": "openai_responses",
    "AnthropicConfig": "anthropic",
    "GeminiConfig": "gemini",
    "OllamaConfig": "ollama",
    "DashScopeConfig": "dashscope",
}

_NON_SERIALIZABLE_FIELDS = frozenset({
    "http_client",
})


def _is_sentinel(value: object) -> bool:
    """Return True if *value* is a provider-SDK sentinel (e.g. OpenAI ``omit`` / ``not_given``)."""
    type_name = type(value).__name__
    return type_name in ("Omit", "NotGiven", "NotGivenOr")


def _is_json_safe(value: object) -> bool:
    """Quick check: can *value* survive a JSON round-trip?"""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


class ConfigSpec(BaseModel):
    """JSON-serializable description of an LLM provider config."""

    provider: str
    """Provider identifier, e.g. ``"openai"``, ``"anthropic"``, ``"gemini"``."""

    model: str
    """Model name, e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Extra serializable parameters (temperature, top_p, max_tokens, ...)."""

    @classmethod
    def from_config(cls, config: ModelConfig) -> ConfigSpec:
        """Extract a ``ConfigSpec`` from a live ``ModelConfig`` dataclass."""
        class_name = type(config).__name__
        provider = _CONFIG_CLASS_TO_PROVIDER.get(class_name)
        if provider is None:
            raise ValueError(f"Unknown config class {class_name!r}. Known: {list(_CONFIG_CLASS_TO_PROVIDER)}")

        model: str | None = None
        params: dict[str, Any] = {}

        for f in dataclasses.fields(config):
            if f.name in _NON_SERIALIZABLE_FIELDS:
                continue

            value = getattr(config, f.name)

            if _is_sentinel(value) or value is None:
                continue

            if f.name == "model":
                model = str(value)
                continue

            if not _is_json_safe(value):
                continue

            params[f.name] = value

        if model is None:
            raise ValueError(f"Config {class_name} has no 'model' field or it is None")

        return cls(provider=provider, model=model, params=params)

    def to_config(self) -> ModelConfig:
        """Reconstruct a ``ModelConfig`` dataclass from this spec.

        Raises ``ImportError`` if the provider's optional dependency is not installed.
        Raises ``ValueError`` if the provider string is unknown.
        """
        config_cls = _import_config_class(self.provider)
        return config_cls(model=self.model, **self.params)


def _import_config_class(provider: str) -> type:
    """Lazily import and return the config dataclass for *provider*."""
    if provider == "openai":
        from autogen.beta.config.openai.config import OpenAIConfig

        return OpenAIConfig
    if provider == "openai_responses":
        from autogen.beta.config.openai.config import OpenAIResponsesConfig

        return OpenAIResponsesConfig
    if provider == "anthropic":
        from autogen.beta.config.anthropic.config import AnthropicConfig

        return AnthropicConfig
    if provider == "gemini":
        from autogen.beta.config.gemini.config import GeminiConfig

        return GeminiConfig
    if provider == "ollama":
        from autogen.beta.config.ollama.config import OllamaConfig

        return OllamaConfig
    if provider == "dashscope":
        from autogen.beta.config.dashscope.config import DashScopeConfig

        return DashScopeConfig

    raise ValueError(
        f"Unknown provider {provider!r}. Known: openai, openai_responses, anthropic, gemini, ollama, dashscope"
    )


class ResponseSchemaSpec(BaseModel):
    """JSON-serializable description of a response schema."""

    name: str
    description: str | None = None
    json_schema: dict[str, Any]

    def to_response_schema(self) -> ResponseProto[str]:
        """Reconstruct a ``RawSchema`` from this spec."""
        from autogen.beta.response.schema import RawSchema

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
    config: ConfigSpec | None = None
    response_schema: ResponseSchemaSpec | None = None
    variables: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_agent(cls, agent: Agent) -> AgentSpec:
        """Create an ``AgentSpec`` from a live ``Agent`` instance.

        Only serializable state is captured. Dynamic prompts, middleware,
        callbacks, and dependencies are dropped.
        """
        from autogen.beta.tools.final.function_tool import FunctionTool

        # Tool names
        tool_names: list[str] = []
        for t in agent.tools:
            if isinstance(t, FunctionTool):
                tool_names.append(t.schema.function.name)

        # Config
        config_spec: ConfigSpec | None = None
        if agent.config is not None:
            config_spec = ConfigSpec.from_config(agent.config)

        # Response schema
        rs_spec: ResponseSchemaSpec | None = None
        rs = agent._response_schema
        if rs is not None and getattr(rs, "json_schema", None) is not None:
            rs_spec = ResponseSchemaSpec(
                name=rs.name,
                description=getattr(rs, "description", None),
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
            config=config_spec,
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
        from autogen.beta.agent import Agent as AgentCls
        from autogen.beta.tools.final.function_tool import FunctionTool

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

        # Config: explicit param > spec > None
        final_config = config
        if final_config is None and self.config is not None:
            final_config = self.config.to_config()

        # Response schema: explicit param > spec > None
        final_rs = response_schema
        if final_rs is None and self.response_schema is not None:
            final_rs = self.response_schema.to_response_schema()

        # Variables: spec + overrides
        final_variables = dict(self.variables)
        if variables:
            final_variables.update(variables)

        return AgentCls(
            name=self.name,
            prompt=list(self.prompt),
            config=final_config,
            tools=resolved_tools,
            middleware=middleware,
            dependencies=dependencies,
            variables=final_variables or None,
            response_schema=final_rs,
        )

    @classmethod
    def to_agent_from_json(
        cls,
        json_data: str | dict[str, Any],
        *,
        available_tools: Iterable[Tool | Callable[..., Any]] = (),
        config: ModelConfig | None = None,
        middleware: Iterable[MiddlewareFactory] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[str, Any] | None = None,
        response_schema: ResponseProto[Any] | type | None = None,
    ) -> Agent:
        """Parse JSON into an ``AgentSpec`` and reconstruct an ``Agent`` in one step.

        Parameters
        ----------
        json_data:
            A JSON string or a dict to parse into an ``AgentSpec``.
        available_tools, config, middleware, dependencies, variables, response_schema:
            Forwarded to :meth:`to_agent`.
        """
        spec = cls.model_validate_json(json_data) if isinstance(json_data, str) else cls.model_validate(json_data)

        return spec.to_agent(
            available_tools=available_tools,
            config=config,
            middleware=middleware,
            dependencies=dependencies,
            variables=variables,
            response_schema=response_schema,
        )
