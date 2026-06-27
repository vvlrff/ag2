# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""The AG2-side ACP client.

``BridgeState`` holds all the behavior (update routing, per-turn text buffer,
mediated ``fs/*``, permission resolution) and is SDK-free so it can be unit
tested without ``acp``. ``make_bridge`` wraps a ``BridgeState`` in a concrete
``acp.Client`` subclass — the object passed to ``spawn_agent_process`` and bound
to the connection for the whole run.
"""

import asyncio
import os
import signal
from typing import TYPE_CHECKING, Any

import acp
from acp import schema

from autogen.beta.events.types import BinaryResult

from .mappers import content_blocks_to_files, map_session_update
from .permissions import resolve_permission_option_id

if TYPE_CHECKING:
    from autogen.beta.context import ConversationContext

    from .config import ACPConfig


def _confine(fs_root: str, path: str) -> str:
    """Resolve ``path`` under ``fs_root``; raise ``PermissionError`` if it escapes."""
    root = os.path.realpath(fs_root)
    full = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise PermissionError(f"path {path!r} escapes fs_root {fs_root!r}")
    return full


def _signal_name(num: int) -> str:
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"SIG{num}"


class _Terminal:
    """One running command, with capped output capture."""

    def __init__(self, proc: asyncio.subprocess.Process, output_byte_limit: int | None) -> None:
        self.proc = proc
        self.limit = output_byte_limit
        self.buf = bytearray()
        self.truncated = False
        self._reader: asyncio.Task | None = None

    def start(self) -> None:
        self._reader = asyncio.ensure_future(self._read())

    async def _read(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            if self.limit is not None and len(self.buf) + len(chunk) > self.limit:
                self.buf.extend(chunk[: max(0, self.limit - len(self.buf))])
                self.truncated = True
            else:
                self.buf.extend(chunk)

    def exit_status(self) -> dict | None:
        rc = self.proc.returncode
        if rc is None:
            return None
        if rc < 0:
            return {"exit_code": None, "signal": _signal_name(-rc)}
        return {"exit_code": rc, "signal": None}

    async def wait(self) -> dict:
        await self.proc.wait()
        if self._reader is not None:
            await self._reader
        return self.exit_status() or {"exit_code": None, "signal": None}

    def kill(self) -> None:
        try:
            if self.proc.returncode is None:
                self.proc.kill()
        except ProcessLookupError:
            pass


class TerminalManager:
    """Run commands on behalf of the agent, confined under ``fs_root``."""

    def __init__(self, fs_root: str) -> None:
        self.fs_root = fs_root
        self._terminals: dict[str, _Terminal] = {}
        self._counter = 0

    def _get(self, terminal_id: str) -> _Terminal:
        return self._terminals[terminal_id]

    async def create(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        output_byte_limit: int | None = None,
    ) -> str:
        work_cwd = _confine(self.fs_root, cwd) if cwd else os.path.realpath(self.fs_root)
        proc = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            cwd=work_cwd,
            env={**os.environ, **env} if env else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        term = _Terminal(proc, output_byte_limit)
        term.start()
        self._counter += 1
        terminal_id = f"term-{self._counter}"
        self._terminals[terminal_id] = term
        return terminal_id

    def output(self, terminal_id: str) -> tuple[str, bool, dict | None]:
        term = self._get(terminal_id)
        return term.buf.decode(errors="replace"), term.truncated, term.exit_status()

    async def wait(self, terminal_id: str) -> dict:
        return await self._get(terminal_id).wait()

    async def kill(self, terminal_id: str) -> None:
        self._get(terminal_id).kill()

    async def release(self, terminal_id: str) -> None:
        term = self._terminals.pop(terminal_id, None)
        if term is not None:
            term.kill()


class BridgeState:
    """Run-scoped state and behavior for the ACP client (SDK-free)."""

    def __init__(self, config: "ACPConfig") -> None:
        self.config = config
        self.context: ConversationContext | None = None  # updated by ACPClient before each turn
        self._turn_parts: list[str] = []
        self._turn_files: list[BinaryResult] = []
        self.terminals = TerminalManager(config.fs_root or config.cwd)

    # --- per-turn accumulation ---

    def begin_turn(self) -> None:
        self._turn_parts = []
        self._turn_files = []

    @property
    def turn_text(self) -> str:
        return "".join(self._turn_parts)

    @property
    def turn_files(self) -> list[BinaryResult]:
        return list(self._turn_files)

    async def handle_update(self, payload: dict) -> None:
        """Route one ``session/update`` payload to the stream + accumulate output."""
        if payload.get("session_update") == "agent_message_chunk":
            content = payload.get("content")
            if isinstance(content, dict):
                if content.get("type") == "text":
                    self._turn_parts.append(content.get("text", ""))
                else:
                    self._turn_files.extend(content_blocks_to_files([content]))
        event = map_session_update(payload)
        if event is not None and self.context is not None:
            await self.context.send(event)

    # --- mediated filesystem ---

    @property
    def _fs_root(self) -> str:
        return self.config.fs_root or self.config.cwd

    def read_text_file(self, path: str, line: int | None = None, limit: int | None = None) -> str:
        full = _confine(self._fs_root, path)
        with open(full, encoding="utf-8") as f:
            lines = f.readlines()
        if line:
            lines = lines[line - 1 :]
        if limit is not None:
            lines = lines[:limit]
        return "".join(lines)

    def write_text_file(self, content: str, path: str) -> None:
        full = _confine(self._fs_root, path)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    # --- permissions ---

    async def resolve_permission(self, options: list[dict], tool_call: dict) -> str | None:
        return await resolve_permission_option_id(self.config.permission_policy, options, tool_call, self.context)


def _dump(model: Any) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else dict(model)


class ACPBridge(acp.Client):
    """Concrete ``acp.Client`` that routes ACP callbacks into a ``BridgeState``.

    ``state`` is set on each instance by ``make_bridge``; the class-level ``None``
    is just a placeholder so the attribute exists before binding.
    """

    state = None  # set on the instance by make_bridge

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        await self.state.handle_update(_dump(update))

    async def request_permission(
        self,
        options: list[schema.PermissionOption],
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        **kwargs: Any,
    ) -> schema.RequestPermissionResponse:
        opts = [_dump(o) for o in options]
        chosen = await self.state.resolve_permission(opts, _dump(tool_call))
        if chosen is None:
            return schema.RequestPermissionResponse(outcome=schema.DeniedOutcome())
        return schema.RequestPermissionResponse(outcome=schema.AllowedOutcome(option_id=chosen, outcome="selected"))

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> schema.ReadTextFileResponse:
        content = self.state.read_text_file(path, line=line, limit=limit)
        return schema.ReadTextFileResponse(content=content)

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> schema.WriteTextFileResponse:
        self.state.write_text_file(content, path)
        return schema.WriteTextFileResponse()

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[schema.EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> schema.CreateTerminalResponse:
        env_map = {e.name: e.value for e in env} if env else None
        terminal_id = await self.state.terminals.create(
            command, args=args, cwd=cwd, env=env_map, output_byte_limit=output_byte_limit
        )
        return schema.CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> schema.TerminalOutputResponse:
        output, truncated, status = self.state.terminals.output(terminal_id)
        return schema.TerminalOutputResponse(
            output=output,
            truncated=truncated,
            exit_status=schema.TerminalExitStatus(**status) if status else None,
        )

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> schema.WaitForTerminalExitResponse:
        status = await self.state.terminals.wait(terminal_id)
        return schema.WaitForTerminalExitResponse(exit_code=status.get("exit_code"), signal=status.get("signal"))

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> schema.KillTerminalResponse:
        await self.state.terminals.kill(terminal_id)
        return schema.KillTerminalResponse()

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> schema.ReleaseTerminalResponse:
        await self.state.terminals.release(terminal_id)
        return schema.ReleaseTerminalResponse()


def make_bridge(config: "ACPConfig") -> acp.Client:
    """Build an ``acp.Client`` bound to a fresh ``BridgeState``.

    Returns the client instance; ``bridge.state`` exposes the ``BridgeState`` so
    ``ACPClient`` can update ``context`` and read per-turn output.
    """
    bridge = ACPBridge()
    bridge.state = BridgeState(config)
    return bridge
