# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx

from autogen.beta.config.openai.config import OpenAIConfig
from autogen.beta.spec import ConfigSpec


def test_config_spec_round_trip() -> None:
    config = OpenAIConfig(model="gpt-4o", temperature=0.7, streaming=True)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "openai"
    assert cs.model == "gpt-4o"
    assert cs.params["temperature"] == 0.7
    assert cs.params["streaming"] is True

    restored = cs.to_config()
    assert type(restored).__name__ == "OpenAIConfig"
    assert restored.model == "gpt-4o"
    assert restored.temperature == 0.7


def test_filters_http_client() -> None:
    config = OpenAIConfig(model="gpt-4o", http_client=httpx.AsyncClient())
    cs = ConfigSpec.from_config(config)

    assert "http_client" not in cs.params


def test_filters_sentinels() -> None:
    config = OpenAIConfig(model="gpt-4o")
    cs = ConfigSpec.from_config(config)

    assert "temperature" not in cs.params or cs.params.get("temperature") is not None


def test_from_agent_does_not_include_config() -> None:
    from autogen.beta import AgentSpec
    from test.beta.conftest import make_agent

    config = OpenAIConfig(model="gpt-4o", temperature=0.5)
    agent = make_agent(config=config)
    spec = AgentSpec.from_agent(agent)

    assert spec.config is None


def test_to_agent_with_config_param() -> None:
    from autogen.beta import AgentSpec
    from test.beta.conftest import add

    spec = AgentSpec(name="test", tool_names=["add"])
    config = OpenAIConfig(model="gpt-4o-mini")

    agent = spec.to_agent(available_tools=[add], config=config)

    assert agent.config is config


def test_to_agent_with_config_from_spec() -> None:
    from autogen.beta import AgentSpec

    spec = AgentSpec(
        name="test",
        config=ConfigSpec(provider="openai", model="gpt-4o"),
    )

    agent = spec.to_agent()
    assert agent.config is not None
    assert type(agent.config).__name__ == "OpenAIConfig"


def test_config_override_beats_spec() -> None:
    from autogen.beta import AgentSpec

    spec = AgentSpec(
        name="test",
        config=ConfigSpec(provider="openai", model="gpt-4o"),
    )

    override = OpenAIConfig(model="gpt-4o-mini")
    agent = spec.to_agent(config=override)

    assert agent.config is override


def test_to_agent_from_json_with_config() -> None:
    from autogen.beta import AgentSpec

    json_str = '{"name": "bot", "tool_names": []}'
    config = OpenAIConfig(model="gpt-4o-mini")

    agent = AgentSpec.to_agent_from_json(json_str, config=config)

    assert agent.config is config
