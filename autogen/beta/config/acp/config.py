# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Config classes for ACP-backed agents.

``ACPConfig`` implements the :class:`~autogen.beta.config.config.ModelConfig`
protocol; ``create()`` returns an ``ACPClient`` that drives the CLI agent over
the Agent Client Protocol. ``ClaudeCodeConfig`` is a thin subclass with Claude
Code launch defaults.
"""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from typing_extensions import Self

if TYPE_CHECKING:
    from autogen.beta.config.client import LLMClient
    from autogen.beta.context import StreamId

    from .session import ACPSession

PermissionPolicy = Literal["ask", "auto", "deny"]


@dataclass(slots=True)
class ACPConfig:
    """Configuration for driving a CLI coding agent over ACP.

    Attributes:
        command: Executable + base args launching the agent in ACP mode,
            e.g. ``["claude-code-acp"]``. The first element is the executable.
        cwd: Workspace root passed to ``session/new``.
        env: Extra environment variables for the subprocess (auth typically
            lives here or is inherited from the parent process).
        model: Agent model selection, when the CLI supports it.
        permission_policy: How to answer ``session/request_permission``:
            ``"ask"`` routes to the agent's ``hitl_hook``/``context.input``,
            ``"auto"`` allows, ``"deny"`` rejects.
        fs_root: Root for mediated ``fs/*`` access (defaults to ``cwd``).
        allow_terminal: Whether to advertise the ACP terminal capability.
        additional_directories: Extra ACP workspace roots.
        startup_timeout: Seconds to allow for subprocess spawn + handshake.
        turn_timeout: Per-prompt-turn timeout in seconds (``None`` = no limit).
        expose_tools: Expose the agent's function ``tools`` to the CLI agent over
            an in-process MCP server. When ``True`` (default) and the agent has at
            least one function tool, an ephemeral localhost MCP endpoint is started
            on the first turn and advertised to the CLI agent. With no function
            tools, no server is started (zero overhead, identical to Phase 1).
    """

    command: list[str] = field(default_factory=list)
    cwd: str = "."
    env: dict[str, str] | None = None
    model: str | None = None
    permission_policy: PermissionPolicy = "ask"
    fs_root: str | None = None
    allow_terminal: bool = True
    additional_directories: list[str] = field(default_factory=list)
    startup_timeout: float = 30.0
    turn_timeout: float | None = None
    expose_tools: bool = True

    # Run-scoped live sessions, keyed by stream id. Not part of identity and not
    # carried by ``copy()`` (a copy is a distinct config with its own sessions).
    _sessions: "dict[StreamId, ACPSession]" = field(init=False, compare=False, repr=False, default_factory=dict)

    def copy(self, /, **overrides: object) -> Self:
        return replace(self, **overrides)

    def create(self) -> "LLMClient":
        from .client import ACPClient

        return ACPClient(self)

    async def aclose(self) -> None:
        """Tear down every live ACP subprocess started from this config."""
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            await session.close()


@dataclass(slots=True)
class ClaudeCodeConfig(ACPConfig):
    """``ACPConfig`` preset for the Claude Code ACP adapter."""

    command: list[str] = field(default_factory=lambda: ["claude-code-acp"])
