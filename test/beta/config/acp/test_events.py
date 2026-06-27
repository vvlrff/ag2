# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.config.acp.events import (
    ACPAvailableCommands,
    ACPModeChange,
    ACPPlan,
    ACPPlanEntry,
)
from autogen.beta.events import BaseEvent


def test_acp_plan_holds_entries():
    plan = ACPPlan(entries=[ACPPlanEntry(content="step 1", status="pending", priority="high")])
    assert isinstance(plan, BaseEvent)
    assert plan.entries[0].content == "step 1"
    assert plan.entries[0].status == "pending"
    assert plan.entries[0].priority == "high"


def test_mode_change():
    assert ACPModeChange(mode_id="edit").mode_id == "edit"


def test_available_commands():
    assert ACPAvailableCommands(commands=["/test"]).commands == ["/test"]
