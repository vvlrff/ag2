# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import sys

import pytest


@pytest.fixture
def fake_agent_command() -> list[str]:
    """Command that launches the in-repo fake ACP agent subprocess."""
    script = pathlib.Path(__file__).parent / "_fake_agent.py"
    return [sys.executable, str(script)]
