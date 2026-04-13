# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.anthropic.config import AnthropicConfig
from autogen.beta.spec import ConfigSpec


def test_config_spec_round_trip() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-20250514", temperature=0.5, max_tokens=1024)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "anthropic"
    assert cs.model == "claude-sonnet-4-20250514"
    assert cs.params["temperature"] == 0.5
    assert cs.params["max_tokens"] == 1024

    restored = cs.to_config()
    assert type(restored).__name__ == "AnthropicConfig"
    assert restored.model == "claude-sonnet-4-20250514"
