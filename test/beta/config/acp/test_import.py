# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


def test_acp_package_imports():
    import autogen.beta.config.acp as pkg

    assert "ACPConfig" in pkg.__all__
    assert "ClaudeCodeConfig" in pkg.__all__
