# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import fnmatch
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from autogen.beta.annotations import Context
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.final.function_tool import FunctionTool, tool
from autogen.beta.tools.tool import Tool

# Commands that only read state and never modify the filesystem.
# Used when ``LocalShellEnvironment.readonly=True`` and no explicit ``allowed``
# list is provided.  This is a best-effort list — commands like ``echo`` can
# still redirect output (``echo x > file``), because ``shell=True`` processing
# happens inside the OS shell after our prefix check.
_READONLY_COMMANDS: tuple[str, ...] = (
    "cat",
    "head",
    "tail",
    "ls",
    "ll",
    "la",
    "grep",
    "egrep",
    "fgrep",
    "find",
    "wc",
    "du",
    "df",
    "diff",
    "stat",
    "file",
    "which",
    "pwd",
    "echo",
    "env",
    "printenv",
    "sort",
    "uniq",
    "cut",
    "git log",
    "git diff",
    "git status",
    "git show",
    "git branch",
)


@dataclass(slots=True)
class LocalShellEnvironment:
    """Working directory and execution policy for :class:`LocalShellTool`.

    Quick start::

        # No restrictions — agent can run any command
        env = LocalShellEnvironment(path="/tmp/my_project")

        # Only allow git and python commands
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            allowed=["git", "python", "pip"],
        )

        # Block dangerous commands
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            blocked=["rm -rf", "curl", "wget"],
        )

        # Hide sensitive files from the agent
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            ignore=["**/.env", "*.key", "secrets/**"],
        )

        # Read-only mode — agent can inspect but not modify
        env = LocalShellEnvironment(
            path="/tmp/my_project",
            readonly=True,
        )

    Args:
        path: Working directory for all shell commands. If ``None``, a temporary
              directory is created automatically with prefix ``"ag2_shell_"``.
        cleanup: Delete the directory on process exit. Defaults to ``True`` when
                 ``path=None`` (auto temp dir) and ``False`` when ``path`` is set.
        allowed: Whitelist of command prefixes. Only commands *starting with* one
                 of these strings are executed. All others return
                 ``"Command not allowed: ..."``

                 Example: ``["git", "python", "uv run"]``
                 → ``"git status"`` ✓, ``"python app.py"`` ✓, ``"rm file"`` ✗

                 Applied before ``blocked``. Takes precedence over ``readonly``.
        blocked: Blacklist of command prefixes. Commands *starting with* any of
                 these strings are rejected with ``"Command not allowed: ..."``

                 Example: ``["rm -rf", "curl", "wget"]``
                 → ``"rm -rf /"`` ✗, ``"curl evil.com"`` ✗, ``"ls"`` ✓

                 Applied after ``allowed``.
        ignore: Gitignore-style path patterns. Literal file paths parsed from the
                command string are resolved and checked against these patterns.
                Matches return ``"Access denied: <path>"``. Paths that resolve
                outside the working directory are always denied.

                Example: ``["**/.env", "*.key", "secrets/**"]``
                - ``"cat .env"`` ✗, ``"cat secrets/db.key"`` ✗, ``"cat app.py"`` ✓

                Only literal paths are checked — dynamically computed paths
                (shell variables, command substitution) are not intercepted.
        readonly: If ``True`` and ``allowed`` is not set, restrict commands to a
                  built-in read-only list (``cat``, ``ls``, ``grep``, ``git log``,
                  etc.). This is **best-effort**: shell redirections such as
                  ``echo x > file`` still bypass the check because evaluation
                  happens inside the OS shell after the prefix check.

                  Set ``allowed`` explicitly if you need a custom read-only list.
        env: Extra environment variables merged into each command's environment.
             These are added *on top of* the current process environment, so
             ``PATH``, ``HOME``, and existing vars remain available.

             Example: ``{"MY_VAR": "value", "DEBUG": "1"}``
        timeout: Per-command timeout in seconds. Timed-out commands return an
                 error string with ``[exit code: 124]`` (Unix convention).
                 Default: 60.
        max_output: Maximum number of characters returned from a command. Output
                    longer than this is truncated and a note is appended so the
                    model knows data was cut. Default: 100 000 (~100 KB).
    """

    path: str | Path | None = None
    cleanup: bool | None = None
    allowed: list[str] | None = None
    blocked: list[str] | None = None
    ignore: list[str] | None = None
    readonly: bool = False
    env: dict[str, str] | None = None
    timeout: float = 60
    max_output: int = 100_000


def _matches(pattern: str, command: str) -> bool:
    """Return True if *command* starts with *pattern* as a whole word or prefix.

    ``"git"`` matches ``"git status"`` and ``"git"`` but not ``"gitconfig"``.
    ``"uv run"`` matches ``"uv run pytest"`` but not ``"uv add requests"``.
    """
    stripped = command.strip()
    if not stripped.startswith(pattern):
        return False
    # Ensure the match is at a word boundary: pattern ends the string or is
    # followed by whitespace.
    rest = stripped[len(pattern) :]
    return rest == "" or rest[0] == " "


def _check_ignore(command: str, workdir: Path, patterns: list[str]) -> str | None:
    """Return ``"Access denied: <path>"`` if any literal path in *command* matches *patterns*.

    Tokens are extracted via :func:`shlex.split` to handle quoted paths. Each
    token is resolved relative to *workdir* and checked against each pattern:

    - ``".env"`` / ``"*.key"`` — matches by filename anywhere in the tree.
    - ``"secrets"`` — matches the file or directory named ``secrets`` and
      everything inside it (``secrets/db.key``, ``secrets/nested/x``).
    - ``"secrets/**"`` / ``"**/.env"`` — explicit glob patterns via fnmatch.

    Paths that resolve outside *workdir* (path traversal, absolute paths) are
    always denied regardless of patterns.

    Returns ``None`` if no pattern matches.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    resolved_workdir = workdir.resolve()

    for token in tokens:
        try:
            resolved = (workdir / token).resolve()
        except Exception:
            continue

        try:
            # Normalize to forward slashes for cross-platform pattern matching.
            rel = str(resolved.relative_to(resolved_workdir)).replace("\\", "/")
        except ValueError:
            # Path resolved outside workdir (e.g. "../../../etc/passwd" or an
            # absolute path).  Block it unconditionally — ignore patterns are
            # meant to protect sensitive files; silently allowing out-of-workdir
            # paths would create a trivial bypass.
            return f"Access denied: {resolved}"

        for pattern in patterns:
            # Glob pattern (contains * ? [) — match relative path and filename.
            if any(c in pattern for c in ("*", "?", "[")):
                if fnmatch.fnmatch(rel, pattern):
                    return f"Access denied: {resolved}"
                # "**/<name>": also match filename directly (file in workdir root)
                if pattern.startswith("**/") and fnmatch.fnmatch(resolved.name, pattern[3:]):
                    return f"Access denied: {resolved}"
                # Simple glob like "*.key": match filename
                if fnmatch.fnmatch(resolved.name, pattern):
                    return f"Access denied: {resolved}"
            else:
                # Plain name (no wildcards): match filename or directory prefix.
                # "secrets" blocks secrets, secrets/db.key, secrets/nested/x
                # ".env" blocks .env anywhere in the tree
                if resolved.name == pattern or rel == pattern or rel.startswith(pattern + "/"):
                    return f"Access denied: {resolved}"

    return None


class LocalShellTool(Tool):
    """Client-side shell execution tool. Works with any LLM provider.

    Unlike provider-native shell tools, this tool executes commands locally via
    ``subprocess`` and is compatible with any :class:`~autogen.beta.config.ModelConfig`
    (OpenAI, Anthropic, Gemini, Ollama, etc.).

    The working directory persists for the lifetime of the tool instance, so
    files created in one ``agent.ask()`` call are visible in subsequent calls.

    Args:
        environment: Execution configuration. Pass ``None`` to use a temporary
                     directory that is cleaned up automatically on exit.

    Examples::

        # 1. Auto temp dir — cleaned up on exit
        tool = LocalShellTool()

        # 2. Persistent directory
        tool = LocalShellTool(environment=LocalShellEnvironment(path="/tmp/my_project"))

        # 3. Only allow safe commands + hide secrets
        tool = LocalShellTool(
            environment=LocalShellEnvironment(
                path="/tmp/my_project",
                allowed=["python", "pip", "git"],
                ignore=["**/.env", "*.key"],
            )
        )

        # 4. Read-only inspection of an existing directory
        tool = LocalShellTool(
            environment=LocalShellEnvironment(
                path="/tmp/my_project",
                readonly=True,
            )
        )
    """

    __slots__ = ("_environment", "_workdir", "_tool")

    def __init__(self, *, environment: LocalShellEnvironment | None = None) -> None:
        if environment is None:
            environment = LocalShellEnvironment()

        self._environment = environment

        # cleanup default: True for auto-tempdir, False for user-specified path
        cleanup = environment.cleanup if environment.cleanup is not None else (environment.path is None)

        if environment.path is None:
            tmpdir = tempfile.mkdtemp(prefix="ag2_shell_")
            self._workdir = Path(tmpdir)
            if cleanup:
                atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        else:
            self._workdir = Path(environment.path).resolve()
            self._workdir.mkdir(parents=True, exist_ok=True)
            if cleanup:
                workdir_str = str(self._workdir)
                atexit.register(lambda: shutil.rmtree(workdir_str, ignore_errors=True))

        workdir = self._workdir
        env = environment

        # readonly=True with no explicit allowed → use built-in read-only list.
        # explicit allowed always takes precedence over readonly.
        effective_allowed: list[str] | None = env.allowed
        if env.readonly and env.allowed is None:
            effective_allowed = list(_READONLY_COMMANDS)

        def _shell(command: str) -> str:
            # 1. allowed — if set, command MUST match at least one pattern
            if effective_allowed is not None and not any(_matches(p, command) for p in effective_allowed):
                return f"Command not allowed: {command!r}"

            # 2. blocked — if any pattern matches, reject
            if env.blocked is not None and any(_matches(p, command) for p in env.blocked):
                return f"Command not allowed: {command!r}"

            # 3. ignore — check literal paths in the command string
            if env.ignore is not None:
                denied = _check_ignore(command, workdir, env.ignore)
                if denied is not None:
                    return denied

            # Merge caller's extra env vars on top of the current process env.
            # Passing env=None to subprocess inherits everything; passing a dict
            # REPLACES the env entirely — so we always merge explicitly.
            merged_env = {**os.environ, **env.env} if env.env is not None else None

            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=env.timeout,
                    env=merged_env,
                )
                output = (result.stdout + result.stderr).strip()

                # Truncate long output so it doesn't flood the model context.
                if len(output) > env.max_output:
                    total = len(output)
                    output = output[: env.max_output]
                    output += f"\n[truncated: showing first {env.max_output} of {total} chars]"

                if result.returncode != 0:
                    suffix = f"[exit code: {result.returncode}]"
                    return f"{output}\n{suffix}" if output else suffix
                return output
            except subprocess.TimeoutExpired:
                return f"Command timed out after {env.timeout}s\n[exit code: 124]"

        self._tool: FunctionTool = tool(
            _shell,
            name="shell",
            description=f"Execute a shell command in the working directory: {workdir}",
        )

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
