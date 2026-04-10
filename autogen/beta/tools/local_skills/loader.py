# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


class SkillNotFoundError(KeyError):
    """Raised when a skill cannot be found in configured paths."""


class InvalidSkillNameError(ValueError):
    """Raised when a skill name is empty or malformed."""


class InvalidSkillError(ValueError):
    """Raised when skill metadata violates the specification."""


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

    Frontmatter parsing and strict validation rules are aligned with:
    https://agentskills.io/specification

    Search priority (first match wins for duplicate names):

    1. ``{cwd}/.agents/skills/``  — project-level, cross-client
    2. ``~/.agents/skills/``      — user-level, cross-client
    3. Any *paths* supplied to the constructor (appended in order)
    """

    DEFAULT_PATHS: list[Path] = [
        Path(".agents/skills"),
        Path.home() / ".agents/skills",
    ]

    _SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

    def __init__(self, *paths: str | Path, strict: bool = True) -> None:
        if paths:
            self._paths = [Path(p) for p in paths]
        else:
            self._paths = list(self.DEFAULT_PATHS)
        self._strict = strict

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
                text = skill_md.read_text(encoding="utf-8")
                fm_raw: dict[str, object] = {}
                # Parse SKILL.md frontmatter as defined by:
                # https://agentskills.io/specification
                if text.startswith("---"):
                    end = text.find("\n---", 3)
                    if end != -1:
                        yaml_block = text[3:end].strip()
                        parsed = yaml.safe_load(yaml_block)
                        if isinstance(parsed, dict):
                            fm_raw = {str(k): v for k, v in parsed.items() if v is not None}
                fm = {k: str(v) for k, v in fm_raw.items()}
                meta = SkillMetadata(
                    name=fm.get("name") or skill_dir.name,
                    description=fm.get("description") or "",
                    path=skill_dir,
                    has_scripts=(skill_dir / "scripts").is_dir(),
                    version=fm.get("version") or None,
                    license=fm.get("license") or None,
                    compatibility=fm.get("compatibility") or None,
                )
                if self._strict:
                    self._validate_skill_metadata(skill_dir, fm_raw, meta)
                if meta.name not in seen:
                    seen[meta.name] = meta
        return sorted(seen.values(), key=lambda m: m.name)

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

    def _find_dir(self, name: str) -> Path:
        if not name.strip():
            raise InvalidSkillNameError("skill name must not be empty")
        if "/" in name or "\\" in name:
            raise InvalidSkillNameError("skill name must not contain path separators")
        for meta in self.discover():
            if meta.name == name:
                return meta.path
        raise SkillNotFoundError(f"Skill {name!r} not found in any configured path")

    @classmethod
    def _validate_skill_metadata(cls, skill_dir: Path, fm: dict[str, object], meta: SkillMetadata) -> None:
        if "name" not in fm:
            raise InvalidSkillError(f"Skill {skill_dir.name!r} is missing required frontmatter field: name")
        if "description" not in fm:
            raise InvalidSkillError(f"Skill {skill_dir.name!r} is missing required frontmatter field: description")

        name = meta.name
        description = meta.description
        compatibility = meta.compatibility

        if not (1 <= len(name) <= 64):
            raise InvalidSkillError(f"Invalid skill name {name!r}: expected length 1-64")
        if not cls._SKILL_NAME_RE.fullmatch(name):
            raise InvalidSkillError(f"Invalid skill name {name!r}: expected lowercase alnum and single hyphens")
        if name != skill_dir.name:
            raise InvalidSkillError(f"Skill name {name!r} must match directory name {skill_dir.name!r}")

        if not (1 <= len(description) <= 1024):
            raise InvalidSkillError(f"Invalid description for {name!r}: expected length 1-1024")

        if compatibility is not None and not (1 <= len(compatibility) <= 500):
            raise InvalidSkillError(f"Invalid compatibility for {name!r}: expected length 1-500")

        metadata = fm.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise InvalidSkillError(f"Invalid metadata for {name!r}: expected mapping")

        allowed_tools = fm.get("allowed-tools")
        if allowed_tools is not None and not isinstance(allowed_tools, str):
            raise InvalidSkillError(f"Invalid allowed-tools for {name!r}: expected string")
