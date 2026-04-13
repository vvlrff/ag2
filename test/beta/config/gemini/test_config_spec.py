# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.gemini.config import GeminiConfig
from autogen.beta.spec import ConfigSpec


def test_config_spec_round_trip() -> None:
    config = GeminiConfig(model="gemini-2.0-flash", temperature=0.3)
    cs = ConfigSpec.from_config(config)

    assert cs.provider == "gemini"
    assert cs.model == "gemini-2.0-flash"
    assert cs.params["temperature"] == 0.3

    restored = cs.to_config()
    assert type(restored).__name__ == "GeminiConfig"
