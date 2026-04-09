# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.shell.environment.local import LocalShellEnvironment

from .loader import SkillLoader


class LocalSkillsTool(Toolkit):
    """Client-side skills toolkit using the agentskills.io convention.

    Implements a three-step progressive-disclosure pattern:

    1. **list_skills()** — returns a lightweight catalog (name + description).
    2. **load_skill(name)** — returns the full ``SKILL.md`` instructions on demand.
    3. **run_skill_script(name, script, args)** — executes a script from the
       skill's ``scripts/`` directory via :class:`~autogen.beta.tools.shell.LocalShellEnvironment`.

    Works with *any* provider (no provider-specific API required).

    Example::

        # Scan default paths (.agents/skills)
        LocalSkillsTool()

        # Scan only the given paths (defaults are NOT included)
        LocalSkillsTool("./skills")
        LocalSkillsTool("/path/to/skills-a", "/path/to/skills-b")
    """

    list_skills: FunctionTool
    load_skill: FunctionTool
    run_skill_script: FunctionTool

    def __init__(self, *paths: str | Path) -> None:
        loader = SkillLoader(*paths)
        self.list_skills = _make_list_tool(loader)
        self.load_skill = _make_load_tool(loader)
        self.run_skill_script = _make_run_tool(loader)

        tools = [self.list_skills, self.load_skill, self.run_skill_script]
        super().__init__(*tools)


def _make_list_tool(loader: SkillLoader) -> FunctionTool:
    def list_skills() -> list[dict[str, str]]:
        """List all available skills with their names and descriptions.

        Call load_skill(name) to get the full instructions for a specific skill.
        """
        return [{"name": m.name, "description": m.description} for m in loader.discover()]

    return tool(list_skills)


def _make_load_tool(loader: SkillLoader) -> FunctionTool:
    def load_skill(name: str) -> str:
        """Load the full SKILL.md instructions for a skill by name."""
        return loader.load(name)

    return tool(load_skill)


def _make_run_tool(loader: SkillLoader) -> FunctionTool:
    def run_skill_script(name: str, script: str, args: list[str] | None = None) -> str:
        """Execute a script from a skill's scripts/ directory.

        Args:
            name:   Skill name (as returned by list_skills).
            script: Script filename inside the skill's ``scripts/`` directory
                    (e.g. ``"scaffold.py"`` or ``"build.sh"``).
            args:   Optional list of arguments passed to the script.

        Returns:
            Combined stdout + stderr output of the script.
        """
        skill_dir = loader.get_path(name)
        scripts_dir = skill_dir / "scripts"
        script_path = scripts_dir / script

        if not script_path.exists():
            return f"Error: script '{script}' not found in {scripts_dir}"

        prefix = "python " if script.endswith(".py") else ""
        # Use just the script filename — cwd is already scripts_dir
        cmd = prefix + script
        if args:
            cmd += " " + " ".join(args)

        env = LocalShellEnvironment(path=scripts_dir, cleanup=False)
        return env.run(cmd)

    return tool(run_skill_script)
