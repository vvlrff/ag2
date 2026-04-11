# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import atexit
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from autogen.beta.tools.local_skills.loader import SkillMetadata
from autogen.beta.tools.shell.environment.base import ShellEnvironment

__all__ = ("LocalRuntime", "SkillRuntime")


@runtime_checkable
class SkillRuntime(Protocol):
    """Unified runtime: storage, discovery, and execution of skills.

    A runtime is responsible for three concerns:

    1. **Storage** — where skills are installed (``install``, ``remove``).
    2. **Discovery** — scanning for installed skills (``discover``, ``load``, ``invalidate``).
    3. **Execution** — providing a shell environment to run scripts (``shell``).

    :class:`LocalRuntime` is the default implementation.  A ``DockerRuntime``
    can be added later without changes to the framework code.
    """

    @property
    def cleanup(self) -> bool:
        """Delete runtime storage on process exit."""
        ...

    @property
    def lock_dir(self) -> Path:
        """Local directory where ``skills-lock.json`` is stored.

        Always a local path — the lock file is host metadata, not runtime storage.
        ``LocalRuntime`` returns ``_install_dir``.  A future ``DockerRuntime``
        would return a configurable local directory.
        """
        ...

    # --- Discovery ---

    def discover(self) -> list[SkillMetadata]:
        """Return metadata for all installed skills."""
        ...

    def load(self, name: str) -> str:
        """Return the full ``SKILL.md`` text for *name*."""
        ...

    def invalidate(self) -> None:
        """Clear discovery cache (call after install / remove)."""
        ...

    # --- Storage ---

    def install(self, source: Path, name: str) -> None:
        """Move an extracted skill from a staging directory into runtime storage.

        Args:
            source: Local staging directory that contains the skill files.
            name:   Skill name (used as the sub-directory name in storage).
        """
        ...

    def remove(self, name: str) -> None:
        """Delete an installed skill from storage.

        Raises:
            ValueError:      If *name* would resolve outside the install directory.
            FileNotFoundError: If no skill with *name* is installed.
        """
        ...

    # --- Execution ---

    def shell(self, scripts_dir: Path) -> ShellEnvironment:
        """Return a :class:`~autogen.beta.tools.shell.ShellEnvironment` for *scripts_dir*.

        Args:
            scripts_dir: Absolute path to the skill's ``scripts/`` directory.
                         Callers resolve this path via the discovery loader so
                         that both install-dir and extra-path skills are handled
                         uniformly.
        """
        ...


@dataclass
class LocalRuntime:
    """Local filesystem storage and subprocess execution.

    Args:
        dir:        Directory where skills are installed.
                    ``None`` → ``.agents/skills`` (default).
        cleanup:    If ``True``, the install directory is deleted at process exit.
        timeout:    Per-command timeout in seconds. Defaults to 60.
        max_output: Maximum characters returned from a script run. Defaults to 100,000.
        blocked:    Command prefixes that are not allowed to run. Empty list → nothing blocked.

    Example::

        # Default — installs to .agents/skills
        SkillSearchToolset()

        # Custom directory
        SkillSearchToolset(runtime=LocalRuntime("./my-skills"))

        # Full control
        SkillSearchToolset(
            runtime=LocalRuntime("./my-skills", timeout=30, cleanup=True, blocked=["rm -rf"]),
        )
    """

    dir: str | Path | None = None
    cleanup: bool = False
    timeout: float = 60
    max_output: int = 100_000
    blocked: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._install_dir = Path(self.dir) if self.dir is not None else Path(".agents/skills")
        self._loader: object | None = None  # SkillLoader, lazy
        if self.cleanup:
            atexit.register(shutil.rmtree, str(self._install_dir), True)

    @property
    def install_dir(self) -> Path:
        """Resolved install directory."""
        return self._install_dir

    @property
    def lock_dir(self) -> Path:
        return self._install_dir

    # --- Discovery ---

    def _get_loader(self) -> object:
        if self._loader is None:
            from autogen.beta.tools.local_skills.loader import SkillLoader

            self._loader = SkillLoader(self._install_dir)
        return self._loader

    def discover(self) -> list[SkillMetadata]:

        return self._get_loader().discover()  # type: ignore[union-attr]

    def load(self, name: str) -> str:
        return self._get_loader().load(name)  # type: ignore[union-attr]

    def invalidate(self) -> None:
        self._get_loader().invalidate()  # type: ignore[union-attr]

    # --- Storage ---

    def install(self, source: Path, name: str) -> None:
        dest = self._install_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

    def remove(self, name: str) -> None:
        target = (self._install_dir / name).resolve()
        if not target.is_relative_to(self._install_dir.resolve()):
            raise ValueError(f"Cannot remove '{name}': path traversal detected")
        if not target.exists():
            raise FileNotFoundError(f"Cannot remove '{name}': skill not found in {self._install_dir}")
        shutil.rmtree(target)

    # --- Execution ---

    def shell(self, scripts_dir: Path) -> ShellEnvironment:
        from autogen.beta.tools.shell.environment.local import LocalShellEnvironment

        return LocalShellEnvironment(
            path=scripts_dir,
            cleanup=False,
            timeout=self.timeout,
            max_output=self.max_output,
            blocked=self.blocked or None,
        )
