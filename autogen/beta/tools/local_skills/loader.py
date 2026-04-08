# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillMetadata:
    """Metadata parsed from a skill's SKILL.md frontmatter."""

    name: str
    description: str
    path: Path
    has_scripts: bool
    version: str | None = None
    license: str | None = None
    compatibility: str | None = None


class SkillLoader:
    """Discovers and loads skills from the filesystem.

    Follows the `agentskills.io <https://agentskills.io>`_ progressive-disclosure
    convention: each skill lives in its own directory that contains a ``SKILL.md``
    file with a YAML frontmatter header.

    Search priority (first match wins for duplicate names):

    1. ``{cwd}/.agents/skills/``  — project-level, cross-client
    2. ``{cwd}/.claude/skills/``  — project-level, Claude-compatible
    3. ``~/.agents/skills/``      — user-level, cross-client
    4. ``~/.claude/skills/``      — user-level, Claude-compatible
    5. Any *extra_paths* supplied to the constructor (appended in order)
    """

    DEFAULT_PATHS: list[Path] = [
        Path(".agents/skills"),
        Path.home() / ".agents/skills",
    ]

    def __init__(
        self,
        *extra_paths: str | Path,
        scan_default: bool = True,
    ) -> None:
        paths: list[Path] = list(self.DEFAULT_PATHS) if scan_default else []
        paths.extend(Path(p) for p in extra_paths)
        self._paths = paths

    def discover(self) -> list[SkillMetadata]:
        """Scan all configured paths and return metadata for every skill found.

        When the same skill name appears in more than one path, the first
        occurrence (higher-priority path) wins.
        """
        seen: dict[str, SkillMetadata] = {}
        for base in self._paths:
            if not base.exists():
                continue
            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                meta = self._parse(skill_dir, skill_md)
                if meta.name not in seen:
                    seen[meta.name] = meta
        return list(seen.values())

    def load(self, name: str) -> str:
        """Return the full text of a skill's ``SKILL.md`` by skill name.

        Raises:
            KeyError: if no skill with that name is found.
        """
        skill_dir = self._find_dir(name)
        return (skill_dir / "SKILL.md").read_text(encoding="utf-8")

    def get_path(self, name: str) -> Path:
        """Return the directory path of a skill by name.

        Raises:
            KeyError: if no skill with that name is found.
        """
        return self._find_dir(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_dir(self, name: str) -> Path:
        for base in self._paths:
            skill_dir = base / name
            if (skill_dir / "SKILL.md").exists():
                return skill_dir
        raise KeyError(f"Skill {name!r} not found in any configured path")

    def _parse(self, skill_dir: Path, skill_md: Path) -> SkillMetadata:
        text = skill_md.read_text(encoding="utf-8")
        fm = self._parse_frontmatter(text)
        return SkillMetadata(
            name=fm.get("name") or skill_dir.name,
            description=fm.get("description") or "",
            path=skill_dir,
            has_scripts=(skill_dir / "scripts").is_dir(),
            version=fm.get("version") or None,
            license=fm.get("license") or None,
            compatibility=fm.get("compatibility") or None,
        )

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, str]:
        """Parse a simple YAML frontmatter block (``--- ... ---``)."""
        if not text.startswith("---"):
            return {}
        end = text.find("\n---", 3)
        if end == -1:
            return {}
        yaml_block = text[3:end].strip()
        result: dict[str, str] = {}
        for line in yaml_block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result
