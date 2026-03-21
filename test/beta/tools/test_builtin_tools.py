# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from autogen.beta.annotations import Variable
from autogen.beta.config.anthropic.mappers import tool_to_api as anthropic_tool_to_api
from autogen.beta.config.openai.mappers import tool_to_responses_api as openai_tool_to_api
from autogen.beta.context import Context
from autogen.beta.exceptions import UnsupportedToolError
from autogen.beta.tools.builtin.memory import MemoryTool, MemoryToolSchema
from autogen.beta.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    LocalEnvironment,
    NetworkPolicy,
    ShellTool,
    ShellToolSchema,
)


def _make_context(**variables: object) -> Context:
    return Context(stream=MagicMock(), variables=variables)


# ── MemoryTool ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_tool_schema_type() -> None:
    tool = MemoryTool()
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, MemoryToolSchema)
    assert schema.type == "memory"


def test_memory_tool_anthropic_mapper() -> None:
    schema = MemoryToolSchema()

    result = anthropic_tool_to_api(schema)

    assert result == {"type": "memory_20250818", "name": "memory"}


# ── ShellTool — schema generation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shell_tool_no_environment() -> None:
    tool = ShellTool()
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, ShellToolSchema)
    assert schema.type == "shell"
    assert schema.environment is None


@pytest.mark.asyncio
async def test_shell_tool_container_auto() -> None:
    tool = ShellTool(environment=ContainerAutoEnvironment())
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema.environment, ContainerAutoEnvironment)
    assert schema.environment.network_policy is None


@pytest.mark.asyncio
async def test_shell_tool_container_auto_with_network_policy() -> None:
    policy = NetworkPolicy(allowed_domains=["example.com", "pypi.org"])
    tool = ShellTool(environment=ContainerAutoEnvironment(network_policy=policy))
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema.environment, ContainerAutoEnvironment)
    assert schema.environment.network_policy is policy


@pytest.mark.asyncio
async def test_shell_tool_container_reference() -> None:
    tool = ShellTool(environment=ContainerReferenceEnvironment(container_id="cntr_abc123"))
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema.environment, ContainerReferenceEnvironment)
    assert schema.environment.container_id == "cntr_abc123"


@pytest.mark.asyncio
async def test_shell_tool_local_environment() -> None:
    tool = ShellTool(environment=LocalEnvironment())
    ctx = _make_context()

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema.environment, LocalEnvironment)


@pytest.mark.asyncio
async def test_shell_tool_environment_from_variable() -> None:
    env = ContainerAutoEnvironment()
    tool = ShellTool(environment=Variable("env"))
    ctx = _make_context(env=env)

    [schema] = await tool.schemas(ctx)

    assert schema.environment is env


@pytest.mark.asyncio
async def test_shell_tool_variable_missing_raises() -> None:
    tool = ShellTool(environment=Variable("env"))
    ctx = _make_context()

    with pytest.raises(KeyError, match="env"):
        await tool.schemas(ctx)


# ── ShellTool — Anthropic mapper ──────────────────────────────────────────────


def test_shell_tool_anthropic_mapper_no_environment() -> None:
    schema = ShellToolSchema()

    result = anthropic_tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}


def test_shell_tool_anthropic_mapper_ignores_environment() -> None:
    # Anthropic maps to bash regardless of the environment field
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = anthropic_tool_to_api(schema)

    assert result == {"type": "bash_20250124", "name": "bash"}


# ── ShellTool — OpenAI mapper ─────────────────────────────────────────────────


def test_shell_tool_openai_mapper_no_environment() -> None:
    schema = ShellToolSchema()

    result = openai_tool_to_api(schema)

    assert result == {"type": "shell"}


def test_shell_tool_openai_mapper_container_auto() -> None:
    schema = ShellToolSchema(environment=ContainerAutoEnvironment())

    result = openai_tool_to_api(schema)

    assert result == {"type": "shell", "environment": {"type": "container_auto"}}


def test_shell_tool_openai_mapper_container_auto_with_network_policy() -> None:
    schema = ShellToolSchema(
        environment=ContainerAutoEnvironment(
            network_policy=NetworkPolicy(allowed_domains=["example.com"])
        )
    )

    result = openai_tool_to_api(schema)

    assert result == {
        "type": "shell",
        "environment": {
            "type": "container_auto",
            "network_policy": {
                "type": "allowlist",
                "allowed_domains": ["example.com"],
            },
        },
    }


def test_shell_tool_openai_mapper_container_reference() -> None:
    schema = ShellToolSchema(
        environment=ContainerReferenceEnvironment(container_id="cntr_xyz")
    )

    result = openai_tool_to_api(schema)

    assert result == {
        "type": "shell",
        "environment": {"type": "container_reference", "container_id": "cntr_xyz"},
    }


def test_shell_tool_openai_mapper_local() -> None:
    schema = ShellToolSchema(environment=LocalEnvironment())

    result = openai_tool_to_api(schema)

    assert result == {"type": "shell", "environment": {"type": "local"}}
