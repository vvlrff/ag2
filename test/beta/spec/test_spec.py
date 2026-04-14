# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict
from pydantic import BaseModel

from autogen.beta import AgentSpec
from autogen.beta.exceptions import ToolResolutionError
from autogen.beta.response import ResponseSchema
from autogen.beta.spec import ResponseSchemaSpec
from autogen.beta.tools import WebSearchTool

from .helpers import add, greet, make_agent, multiply


class Answer(BaseModel):
    value: int
    reasoning: str


def test_from_agent_extracts_name_and_prompt() -> None:
    agent = make_agent(prompt="Be helpful.")
    spec = AgentSpec.from_agent(agent)

    assert spec.model_dump() == IsPartialDict({"name": "test_agent", "prompt": ["Be helpful."]})


def test_from_agent_extracts_tools() -> None:
    agent = make_agent(tools=[add, multiply, greet])
    spec = AgentSpec.from_agent(agent)

    assert spec.tool_names == ["add", "multiply", "greet"]


def test_from_agent_no_tools() -> None:
    agent = make_agent(tools=[])
    spec = AgentSpec.from_agent(agent)

    assert spec.tool_names == []


def test_to_agent_resolves_tools() -> None:
    spec = AgentSpec(
        name="resolver",
        prompt=["Resolve tools."],
        tool_names=["add", "greet"],
    )

    agent = spec.to_agent(available_tools=[add, multiply, greet])

    assert agent.name == "resolver"
    assert [t.schema.function.name for t in agent.tools] == ["add", "greet"]


def test_to_agent_missing_tool_raises() -> None:
    spec = AgentSpec(
        name="broken",
        tool_names=["add", "nonexistent"],
    )

    with pytest.raises(ToolResolutionError, match="nonexistent"):
        spec.to_agent(available_tools=[add, multiply])


def test_round_trip_json() -> None:
    agent = make_agent(prompt="Round trip test.")
    spec = AgentSpec.from_agent(agent)
    json_str = spec.model_dump_json()

    restored_spec = AgentSpec.model_validate_json(json_str)
    assert restored_spec == spec

    restored = restored_spec.to_agent(available_tools=[add, multiply])
    assert restored.name == "test_agent"
    assert restored._system_prompt == ["Round trip test."]
    assert len(restored.tools) == 2


def test_manual_spec_creation() -> None:
    spec = AgentSpec(
        name="research_bot",
        prompt=["You are a researcher.", "Be thorough."],
        tool_names=["add"],
    )

    agent = spec.to_agent(available_tools=[add, multiply])

    assert agent.name == "research_bot"
    assert agent._system_prompt == ["You are a researcher.", "Be thorough."]
    assert len(agent.tools) == 1


def test_agent_to_spec_method() -> None:
    agent = make_agent()

    assert agent.to_spec() == AgentSpec.from_agent(agent)


def test_response_schema_spec_round_trip() -> None:
    rs = ResponseSchema(Answer)
    rs_spec = ResponseSchemaSpec(
        name=rs.name,
        description=rs.description,
        json_schema=rs.json_schema,
    )

    json_data = rs_spec.model_dump_json()
    restored = ResponseSchemaSpec.model_validate_json(json_data)

    assert restored.model_dump() == IsPartialDict({"name": rs_spec.name})
    assert restored.json_schema == rs_spec.json_schema

    raw = restored.to_response_schema()
    assert raw.name == rs.name
    assert raw.json_schema == rs.json_schema


def test_from_agent_with_response_schema() -> None:
    agent = make_agent(response_schema=Answer)
    spec = AgentSpec.from_agent(agent)

    assert spec.response_schema is not None
    assert spec.response_schema.name == "Answer"
    assert "value" in str(spec.response_schema.json_schema)


def test_from_json_string() -> None:
    json_str = '{"name": "bot", "prompt": ["Help."], "tool_names": ["add"]}'

    agent = AgentSpec.model_validate_json(json_str).to_agent(available_tools=[add, multiply])

    assert agent.name == "bot"
    assert agent._system_prompt == ["Help."]
    assert [t.schema.function.name for t in agent.tools] == ["add"]


def test_from_dict() -> None:
    data = {
        "name": "bot",
        "prompt": ["Help."],
        "tool_names": ["greet"],
    }

    agent = AgentSpec(**data).to_agent(available_tools=[add, multiply, greet])

    assert agent.name == "bot"
    assert [t.schema.function.name for t in agent.tools] == ["greet"]


def test_from_json_missing_tool_raises() -> None:
    json_str = '{"name": "bot", "tool_names": ["nonexistent"]}'

    with pytest.raises(ToolResolutionError, match="nonexistent"):
        AgentSpec.model_validate_json(json_str).to_agent(available_tools=[add])


def test_json_round_trip() -> None:
    agent = make_agent()
    spec = agent.to_spec()
    json_str = spec.model_dump_json()

    restored = AgentSpec.model_validate_json(json_str).to_agent(available_tools=[add, multiply])

    assert restored.name == agent.name
    assert restored._system_prompt == agent._system_prompt
    assert len(restored.tools) == len(agent.tools)


def test_builtin_tool_serialization() -> None:
    ws = WebSearchTool()
    agent = make_agent(tools=[add, ws])
    spec = AgentSpec.from_agent(agent)

    assert spec.tool_names == ["add", "web_search"]


def test_builtin_tool_resolution() -> None:
    ws = WebSearchTool()
    spec = AgentSpec(
        name="bot",
        tool_names=["add", "web_search"],
    )

    agent = spec.to_agent(available_tools=[add, ws])

    assert len(agent.tools) == 2


def test_missing_builtin_tool_raises() -> None:
    spec = AgentSpec(
        name="bot",
        tool_names=["web_search"],
    )

    with pytest.raises(ToolResolutionError, match="web_search"):
        spec.to_agent(available_tools=[add])


def test_to_agent_passes_hitl_hook() -> None:
    spec = AgentSpec(name="bot")

    def my_hook(msg: str) -> str:
        return msg

    agent = spec.to_agent(hitl_hook=my_hook)
    assert agent._hitl_hook is not None


def test_to_agent_passes_variables() -> None:
    spec = AgentSpec(name="bot")

    agent = spec.to_agent(variables={"key": "value"})
    assert agent._agent_variables == {"key": "value"}


def test_to_agent_passes_dependencies() -> None:
    spec = AgentSpec(name="bot")

    agent = spec.to_agent(dependencies={"db": "mock"})
    assert agent._agent_dependencies == {"db": "mock"}
