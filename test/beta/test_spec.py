# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from pydantic import BaseModel

from autogen.beta import AgentSpec
from autogen.beta.config.anthropic.config import AnthropicConfig
from autogen.beta.config.gemini.config import GeminiConfig
from autogen.beta.config.openai.config import OpenAIConfig
from autogen.beta.response import ResponseSchema
from autogen.beta.spec import ConfigSpec, ResponseSchemaSpec

from .conftest import add, greet, make_agent, multiply


class Answer(BaseModel):
    value: int
    reasoning: str


def test_from_agent_extracts_name_and_prompt() -> None:
    agent = make_agent(prompt="Be helpful.")
    spec = AgentSpec.from_agent(agent)

    assert spec.name == "test_agent"
    assert spec.prompt == ["Be helpful."]


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
    assert len(agent.tools) == 2
    tool_names = [t.schema.function.name for t in agent.tools]
    assert tool_names == ["add", "greet"]


def test_to_agent_missing_tool_raises() -> None:
    spec = AgentSpec(
        name="broken",
        tool_names=["add", "nonexistent"],
    )

    with pytest.raises(ValueError, match="nonexistent"):
        spec.to_agent(available_tools=[add, multiply])


def test_round_trip_json() -> None:
    agent = make_agent(
        prompt="Round trip test.",
        variables={"lang": "en", "count": 42},
    )
    spec = AgentSpec.from_agent(agent)
    json_str = spec.model_dump_json()

    restored_spec = AgentSpec.model_validate_json(json_str)
    assert restored_spec == spec

    restored_agent = restored_spec.to_agent(available_tools=[add, multiply])
    assert restored_agent.name == "test_agent"
    assert restored_agent._system_prompt == ["Round trip test."]
    assert len(restored_agent.tools) == 2


def test_manual_spec_creation() -> None:
    spec = AgentSpec(
        name="research_bot",
        prompt=["You are a researcher.", "Be thorough."],
        tool_names=["add"],
        variables={"max_results": 10},
    )

    agent = spec.to_agent(available_tools=[add, multiply])

    assert agent.name == "research_bot"
    assert agent._system_prompt == ["You are a researcher.", "Be thorough."]
    assert len(agent.tools) == 1
    assert agent._agent_variables == {"max_results": 10}


def test_agent_to_spec_method() -> None:
    agent = make_agent()

    spec_from_method = agent.to_spec()
    spec_from_class = AgentSpec.from_agent(agent)

    assert spec_from_method == spec_from_class


def test_variables_round_trip() -> None:
    agent = make_agent(variables={"key": "value", "num": 123, "flag": True})
    spec = AgentSpec.from_agent(agent)

    assert spec.variables == {"key": "value", "num": 123, "flag": True}

    restored = spec.to_agent(available_tools=[add, multiply])
    assert restored._agent_variables == {"key": "value", "num": 123, "flag": True}


def test_non_serializable_variables_skipped() -> None:
    agent = make_agent(
        variables={
            "safe": "value",
            "unsafe": lambda x: x,  # not JSON-serializable
            "also_safe": 42,
        }
    )
    spec = AgentSpec.from_agent(agent)

    assert "safe" in spec.variables
    assert "also_safe" in spec.variables
    assert "unsafe" not in spec.variables


def test_variables_override_in_to_agent() -> None:
    spec = AgentSpec(
        name="test",
        variables={"a": 1, "b": 2},
    )

    agent = spec.to_agent(variables={"b": 99, "c": 3})

    assert agent._agent_variables == {"a": 1, "b": 99, "c": 3}


def test_response_schema_spec_round_trip() -> None:
    rs = ResponseSchema(Answer)
    rs_spec = ResponseSchemaSpec(
        name=rs.name,
        description=rs.description,
        json_schema=rs.json_schema,
    )

    json_data = rs_spec.model_dump_json()
    restored = ResponseSchemaSpec.model_validate_json(json_data)

    assert restored.name == rs_spec.name
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


def test_to_agent_with_no_config() -> None:
    spec = AgentSpec(name="no_config")
    agent = spec.to_agent()

    assert agent.config is None


def test_config_spec_openai() -> None:
    config = OpenAIConfig(model="gpt-4o", temperature=0.7, streaming=True)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "openai"
    assert cs.model == "gpt-4o"
    assert cs.params["temperature"] == 0.7
    assert cs.params["streaming"] is True

    restored = cs.to_config()
    assert type(restored).__name__ == "OpenAIConfig"
    assert restored.model == "gpt-4o"  # type: ignore[attr-defined]
    assert restored.temperature == 0.7  # type: ignore[attr-defined]


def test_config_spec_anthropic() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-20250514", temperature=0.5, max_tokens=1024)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "anthropic"
    assert cs.model == "claude-sonnet-4-20250514"
    assert cs.params["temperature"] == 0.5
    assert cs.params["max_tokens"] == 1024

    restored = cs.to_config()
    assert type(restored).__name__ == "AnthropicConfig"
    assert restored.model == "claude-sonnet-4-20250514"  # type: ignore[attr-defined]


def test_config_spec_gemini() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", temperature=0.3)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "gemini"
    assert cs.model == "gemini-2.0-flash"
    assert cs.params["temperature"] == 0.3

    restored = cs.to_config()
    assert type(restored).__name__ == "GeminiConfig"


def test_config_spec_filters_http_client() -> None:
    config = OpenAIConfig(model="gpt-4o", http_client=httpx.AsyncClient())
    cs = ConfigSpec.from_config(config)

    assert "http_client" not in cs.params


def test_config_spec_filters_sentinels() -> None:
    config = OpenAIConfig(model="gpt-4o")
    cs = ConfigSpec.from_config(config)

    # Sentinel fields like temperature (default=omit) should not appear
    assert "temperature" not in cs.params or cs.params.get("temperature") is not None


def test_config_spec_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        ConfigSpec(provider="unknown", model="x").to_config()


def test_from_agent_does_not_include_config() -> None:
    config = OpenAIConfig(model="gpt-4o", temperature=0.5)
    agent = make_agent(config=config)
    spec = AgentSpec.from_agent(agent)

    assert spec.config is None


def test_to_agent_with_config_param() -> None:
    spec = AgentSpec(name="test", tool_names=["add"])
    config = OpenAIConfig(model="gpt-4o-mini")

    agent = spec.to_agent(available_tools=[add], config=config)

    assert agent.config is config


def test_to_agent_with_config_from_spec() -> None:
    spec = AgentSpec(
        name="test",
        config=ConfigSpec(provider="openai", model="gpt-4o"),
    )

    agent = spec.to_agent()
    assert agent.config is not None
    assert type(agent.config).__name__ == "OpenAIConfig"


def test_config_override_beats_spec() -> None:
    spec = AgentSpec(
        name="test",
        config=ConfigSpec(provider="openai", model="gpt-4o"),
    )

    override = OpenAIConfig(model="gpt-4o-mini")
    agent = spec.to_agent(config=override)

    assert agent.config is override


def test_to_agent_from_json_string() -> None:
    json_str = '{"name": "bot", "prompt": ["Help."], "tool_names": ["add"]}'

    agent = AgentSpec.to_agent_from_json(json_str, available_tools=[add, multiply])

    assert agent.name == "bot"
    assert agent._system_prompt == ["Help."]
    assert len(agent.tools) == 1
    assert agent.tools[0].schema.function.name == "add"


def test_to_agent_from_json_dict() -> None:
    data = {
        "name": "bot",
        "prompt": ["Help."],
        "tool_names": ["greet"],
        "variables": {"lang": "en"},
    }

    agent = AgentSpec.to_agent_from_json(data, available_tools=[add, multiply, greet])

    assert agent.name == "bot"
    assert len(agent.tools) == 1
    assert agent.tools[0].schema.function.name == "greet"
    assert agent._agent_variables == {"lang": "en"}


def test_to_agent_from_json_with_config_override() -> None:
    json_str = '{"name": "bot", "tool_names": []}'
    config = OpenAIConfig(model="gpt-4o-mini")

    agent = AgentSpec.to_agent_from_json(json_str, config=config)

    assert agent.config is config


def test_to_agent_from_json_missing_tool_raises() -> None:
    json_str = '{"name": "bot", "tool_names": ["nonexistent"]}'

    with pytest.raises(ValueError, match="nonexistent"):
        AgentSpec.to_agent_from_json(json_str, available_tools=[add])


def test_to_agent_from_json_round_trip() -> None:
    agent = make_agent(variables={"x": 1})
    json_str = agent.to_spec().model_dump_json()

    restored = AgentSpec.to_agent_from_json(json_str, available_tools=[add, multiply])

    assert restored.name == agent.name
    assert restored._system_prompt == agent._system_prompt
    assert len(restored.tools) == len(agent.tools)
    assert restored._agent_variables == {"x": 1}
