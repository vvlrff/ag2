# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import fnmatch
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import bashlex
import bashlex.errors

# Commands that only read state and never modify the filesystem.
# Used when ``LocalShellEnvironment.readonly=True`` and no explicit ``allowed``
# list is provided. Per-segment validation in :func:`validate_command` makes
# this list safe against shell operators — e.g. ``echo > file`` is rejected
# in pure read-only mode because redirections are blocked there explicitly.
READONLY_COMMANDS: tuple[str, ...] = (
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


def matches(pattern: str, command: str) -> bool:
    """Return True if *command* starts with *pattern* as a whole word or prefix.

    ``"git"`` matches ``"git status"`` and ``"git"`` but not ``"gitconfig"``.
    ``"uv run"`` matches ``"uv run pytest"`` but not ``"uv add requests"``.
    """
    stripped = command.strip()
    if not stripped.startswith(pattern):
        return False
    rest = stripped[len(pattern) :]
    return rest == "" or rest[0] == " "


def validate_command(
    command: str,
    *,
    allowed: list[str] | None,
    blocked: list[str] | None,
    forbid_redirects: bool,
) -> str | None:
    """Validate *command* against allowed/blocked policy and return ``None``
    when execution is permitted, otherwise a human-readable rejection string.

    The validator parses *command* with :mod:`bashlex` and walks the AST so
    that quoting, pipelines and chaining operators are handled correctly:

    * Pipeline (``|``) and list (``&&``, ``||``, ``;``) operators split the
      command into segments. Each segment must satisfy *allowed* / *blocked*
      independently — ``git log | grep err`` is accepted only when both
      ``git`` and ``grep`` are whitelisted.
    * Command substitution (``$(...)`` / backticks) and process substitution
      (``<(...)`` / ``>(...)``) are rejected when *allowed* is set: their
      contents are too dynamic to validate recursively.
    * Compound constructs (``if``, ``for``, ``while``, ``case``, brace
      groups) are rejected when *allowed* is set, for the same reason.
    * If *forbid_redirects* is ``True``, any ``>``/``>>``/``<``/``<<`` node
      causes rejection. Used to enforce read-only mode where writing to a
      file via redirection would otherwise bypass the read-only intent.
    * Quoted separators (e.g. ``echo "a; b"``) stay inside their argument
      and do not split the command — bashlex handles this for us.
    * If bashlex fails to parse the command, the validator fails closed —
      it is safer to refuse a malformed command than to let an unparsable
      string fall through to the shell.
    """
    if allowed is None and blocked is None and not forbid_redirects:
        return None

    stripped = command.strip()
    if not stripped:
        return None

    try:
        trees = bashlex.parse(command)
    except bashlex.errors.ParsingError:
        return f"Command not allowed (failed to parse shell syntax): {command!r}"
    except (NotImplementedError, IndexError, TypeError):
        return f"Command not allowed (unsupported shell construct): {command!r}"

    for tree in trees:
        rejection = _validate_node(
            tree,
            allowed=allowed,
            blocked=blocked,
            forbid_redirects=forbid_redirects,
            original=command,
        )
        if rejection is not None:
            return rejection
    return None


def _validate_node(
    node: object,
    *,
    allowed: list[str] | None,
    blocked: list[str] | None,
    forbid_redirects: bool,
    original: str,
) -> str | None:
    """Recursively validate a bashlex AST node."""
    kind = getattr(node, "kind", None)

    if kind == "list":
        for child in getattr(node, "parts", ()):
            if getattr(child, "kind", None) == "operator":
                continue
            rejection = _validate_node(
                child,
                allowed=allowed,
                blocked=blocked,
                forbid_redirects=forbid_redirects,
                original=original,
            )
            if rejection is not None:
                return rejection
        return None

    if kind == "pipeline":
        for child in getattr(node, "parts", ()):
            if getattr(child, "kind", None) == "pipe":
                continue
            rejection = _validate_node(
                child,
                allowed=allowed,
                blocked=blocked,
                forbid_redirects=forbid_redirects,
                original=original,
            )
            if rejection is not None:
                return rejection
        return None

    if kind == "compound":
        if allowed is not None:
            return f"Command not allowed (compound commands are not permitted in restricted mode): {original!r}"
        return None

    if kind == "command":
        return _validate_command_segment(
            node,
            allowed=allowed,
            blocked=blocked,
            forbid_redirects=forbid_redirects,
            original=original,
        )

    if allowed is not None:
        return f"Command not allowed (unsupported shell construct {kind!r}): {original!r}"
    return None


def _validate_command_segment(
    node: object,
    *,
    allowed: list[str] | None,
    blocked: list[str] | None,
    forbid_redirects: bool,
    original: str,
) -> str | None:
    """Validate a single ``command``-kind AST node."""
    parts = getattr(node, "parts", ())
    word_positions: list[tuple[int, int]] = []

    for part in parts:
        pkind = getattr(part, "kind", None)
        if pkind == "redirect":
            if forbid_redirects:
                return f"Command not allowed (redirection is not permitted in read-only mode): {original!r}"
            continue
        if pkind == "assignment":
            continue
        if pkind == "word":
            if allowed is not None:
                substitution_kind = _find_substitution_kind(getattr(part, "parts", ()) or ())
                if substitution_kind is not None:
                    return (
                        f"Command not allowed ({substitution_kind} is not permitted in restricted mode): {original!r}"
                    )
            pos = getattr(part, "pos", None)
            if pos is not None:
                word_positions.append(pos)

    if not word_positions:
        return None

    start = word_positions[0][0]
    end = word_positions[-1][1]
    segment = original[start:end]

    if allowed is not None and not any(matches(p, segment) for p in allowed):
        return f"Command not allowed: {segment!r}"
    if blocked is not None and any(matches(p, segment) for p in blocked):
        return f"Command not allowed: {segment!r}"
    return None


def _find_substitution_kind(parts: Iterable[object]) -> str | None:
    """Return a human label for the first substitution node found, else ``None``."""
    for part in parts:
        kind = getattr(part, "kind", None)
        if kind == "commandsubstitution":
            return "command substitution"
        if kind == "processsubstitution":
            return "process substitution"
        nested = getattr(part, "parts", None)
        if nested:
            label = _find_substitution_kind(nested)
            if label is not None:
                return label
    return None


def check_ignore(command: str, workdir: Path, patterns: list[str]) -> str | None:
    """Return ``"Access denied: <path>"`` if any literal path in *command* matches *patterns*.

    Tokens are extracted via :func:`shlex.split` to handle quoted paths. Each
    token is resolved relative to *workdir* and checked against each pattern.

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
            rel = str(resolved.relative_to(resolved_workdir)).replace("\\", "/")
        except ValueError:
            return f"Access denied: {resolved}"

        for pattern in patterns:
            if any(c in pattern for c in ("*", "?", "[")):
                if fnmatch.fnmatch(rel, pattern):
                    return f"Access denied: {resolved}"
                if pattern.startswith("**/") and fnmatch.fnmatch(resolved.name, pattern[3:]):
                    return f"Access denied: {resolved}"
                if fnmatch.fnmatch(resolved.name, pattern):
                    return f"Access denied: {resolved}"
            else:
                if resolved.name == pattern or rel == pattern or rel.startswith(pattern + "/"):
                    return f"Access denied: {resolved}"

    return None


@runtime_checkable
class ShellEnvironment(Protocol):
    @property
    def workdir(self) -> Path: ...

    def run(self, command: str) -> str: ...
