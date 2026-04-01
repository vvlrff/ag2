# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.tool import Tool

from .function_tool import FunctionTool, tool


@dataclass(slots=True)
class LocalShellEnvironment:
    """Working directory configuration for :class:`LocalShellTool`.

    Args:
        path: Directory to use as the shell working directory. If ``None``,
              a temporary directory is created automatically.
        cleanup: Whether to delete the directory when the tool is garbage-
                 collected. Defaults to ``True`` when ``path`` is ``None``
                 (auto temp dir) and ``False`` when ``path`` is provided.
    """

    path: str | Path | None = None
    cleanup: bool | None = None


class LocalShellTool(Tool):
    """Client-side shell execution tool. Works with any LLM provider.

    Unlike :class:`~autogen.beta.tools.ShellTool` (provider schema), this tool
    executes shell commands locally using ``subprocess``. The working directory
    lifecycle is controlled by the ``environment`` parameter.

    Args:
        environment: Directory configuration. If ``None``, defaults to
                     :class:`LocalShellEnvironment` with a temporary directory
                     that is cleaned up on exit.

    """

    __slots__ = ("_environment", "_workdir", "_tempdir", "_tool")

    def __init__(self, *, environment: LocalShellEnvironment | None = None) -> None:
        if environment is None:
            environment = LocalShellEnvironment()

        self._environment = environment
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None

        # cleanup default: True for auto-tempdir, False for user-specified path
        cleanup = environment.cleanup if environment.cleanup is not None else (environment.path is None)

        if environment.path is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="ag2_shell_", delete=cleanup)
            self._workdir = Path(self._tempdir.name)
        else:
            self._workdir = Path(environment.path).resolve()
            self._workdir.mkdir(parents=True, exist_ok=True)
            if cleanup:
                workdir = self._workdir
                atexit.register(lambda: shutil.rmtree(workdir, ignore_errors=True))

        workdir = self._workdir

        @tool
        def shell(command: str) -> str:
            """Execute a shell command in the working directory."""
            result = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return (result.stdout + result.stderr).strip()

        self._tool: FunctionTool = shell

    @property
    def workdir(self) -> Path:
        """The working directory used for command execution."""
        return self._workdir

    async def schemas(self, context: "Context") -> list:
        return await self._tool.schemas(context)

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        self._tool.register(stack, context, middleware=middleware)
