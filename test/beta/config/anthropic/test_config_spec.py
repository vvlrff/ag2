# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dirty_equals import IsPartialDict

from autogen.beta.config.anthropic.config import AnthropicConfig
from autogen.beta.spec import ConfigSpec


def test_config_spec_round_trip() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-20250514", temperature=0.5, max_tokens=1024)
    cs = ConfigSpec.from_config(config)

    assert cs.model_dump() == IsPartialDict({"provider": "anthropic", "model": "claude-sonnet-4-20250514"})
    assert cs.params == IsPartialDict({"temperature": 0.5, "max_tokens": 1024})

    restored = cs.to_config()
    assert isinstance(restored, AnthropicConfig)
    assert restored.model == "claude-sonnet-4-20250514"
    assert restored.temperature == 0.5
