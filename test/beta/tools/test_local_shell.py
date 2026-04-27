# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools import LocalShellTool
from autogen.beta.tools.shell import LocalShellEnvironment
from autogen.beta.tools.shell.environment.base import check_ignore, matches, validate_command


class TestMatches:
    def test_plain_prefix_matches(self) -> None:
        assert matches("git", "git status") is True

    def test_plain_prefix_no_match(self) -> None:
        assert matches("git", "rm -rf /") is False

    def test_multi_word_prefix(self) -> None:
        assert matches("uv run", "uv run pytest") is True
        assert matches("uv run", "uv add requests") is False

    def test_rm_rf_blocked(self) -> None:
        assert matches("rm -rf", "rm -rf /") is True
        assert matches("rm -rf", "rm file.txt") is False

    def test_leading_whitespace_stripped(self) -> None:
        assert matches("git", "  git status") is True

    def test_exact_command_matches(self) -> None:
        # "git" alone (no args) should match
        assert matches("git", "git") is True

    def test_word_boundary_no_false_positive(self) -> None:
        # "git" must not match "gitconfig" or "gitfoo"
        assert matches("git", "gitconfig --list") is False
        assert matches("cat", "catchphrase") is False
        assert matches("py", "python3 app.py") is False


class TestCheckIgnore:
    def test_env_file_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat .env", tmp_path, ["**/.env"])
        assert result is not None
        assert ".env" in result

    def test_key_file_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat server.key", tmp_path, ["*.key"])
        assert result is not None
        assert "server.key" in result

    def test_secrets_dir_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat secrets/db.key", tmp_path, ["secrets/**"])
        assert result is not None
        assert "secrets" in result

    def test_safe_file_allowed(self, tmp_path: Path) -> None:
        result = check_ignore("cat app.py", tmp_path, ["**/.env", "*.key"])
        assert result is None

    def test_quoted_path_handled(self, tmp_path: Path) -> None:
        result = check_ignore('cat ".env"', tmp_path, ["**/.env"])
        assert result is not None

    def test_plain_filename_blocked(self, tmp_path: Path) -> None:
        assert check_ignore("cat .env", tmp_path, [".env"]) is not None

    def test_plain_dirname_blocks_contents(self, tmp_path: Path) -> None:
        assert check_ignore("cat secrets/db.key", tmp_path, ["secrets"]) is not None
        assert check_ignore("cat secrets/nested/x.txt", tmp_path, ["secrets"]) is not None
        assert check_ignore("cat config/prod.yaml", tmp_path, ["secrets"]) is None

    def test_no_patterns_returns_none(self, tmp_path: Path) -> None:
        assert check_ignore("cat .env", tmp_path, []) is None

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        # ../../../etc/passwd resolves outside workdir — must be denied
        result = check_ignore("cat ../../../etc/passwd", tmp_path, ["**/.env"])
        assert result is not None
        assert "Access denied" in result

    def test_absolute_path_outside_workdir_blocked(self, tmp_path: Path) -> None:
        # Absolute path outside workdir must be denied regardless of patterns
        result = check_ignore("cat /etc/passwd", tmp_path, ["**/.env"])
        assert result is not None
        assert "Access denied" in result


class TestLocalShellToolConstruction:
    def test_auto_tempdir_created(self) -> None:
        shell = LocalShellTool()
        assert shell.workdir.exists()
        assert shell.workdir.is_dir()

    def test_explicit_path_created(self, tmp_path: Path) -> None:
        target = tmp_path / "workspace"
        shell = LocalShellTool(environment=target)
        assert shell.workdir == target
        assert target.exists()

    def test_workdir_is_readonly_property(self, tmp_path: Path) -> None:
        shell = LocalShellTool(environment=tmp_path)
        with pytest.raises(AttributeError):
            shell.workdir = tmp_path  # type: ignore[misc]


class TestShellExecution:
    """These tests call the tool function directly via the agent + TestConfig."""

    def _make_tool_call(self, command: str) -> ToolCallEvent:
        return ToolCallEvent(
            arguments=json.dumps({"command": command}),
            name="run_shell_command",
        )

    def _make_config(self, command: str, final_reply: str = "done") -> TestConfig:
        return TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent([self._make_tool_call(command)])),
            final_reply,
        )

    @pytest.mark.asyncio
    async def test_allowed_permits_matching_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, allowed=["echo"]))
        # `echo` is whitelisted; redirection is allowed when the user provided
        # an explicit `allowed=` list (read-only redirect blocking only kicks
        # in for pure `readonly=True` without an explicit allow-list).
        agent = Agent("a", config=self._make_config(f"echo hello > {output}"), tools=[shell])
        await agent.ask("run it")
        assert output.exists(), "echo was allowed but file was not created"
        assert output.read_text().strip() == "hello"

    @pytest.mark.asyncio
    async def test_allowed_blocks_non_matching_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, allowed=["echo"]))
        # "touch" is not in allowed — the file must NOT be created
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    # ── blocked ───────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_blocked_rejects_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, blocked=["touch"]))
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    # ── env merging ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_env_merged_not_replaced(self, tmp_path: Path) -> None:
        """Extra env vars must be added on top of os.environ, not replace it."""
        # Write a helper script — avoids shell variable syntax differences
        # between bash ($VAR) and cmd.exe (%VAR%) across platforms.
        script = tmp_path / "check_env.py"
        script.write_text(
            "import os\n"
            "custom = os.environ.get('MY_CUSTOM_VAR', 'MISSING')\n"
            "path = os.environ.get('PATH', '')\n"
            "print(custom + '|' + path)\n"
        )
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                env={"MY_CUSTOM_VAR": "hello"},
            )
        )
        cmd = f'"{sys.executable}" check_env.py'
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        agent = Agent("a", config=self._make_config(cmd), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        result = tool_results[0]
        assert "hello|" in result, f"MY_CUSTOM_VAR not set: {result!r}"
        path_part = result.split("|", 1)[1] if "|" in result else ""
        assert path_part.strip(), f"PATH was lost — env was replaced instead of merged: {result!r}"

    # ── timeout ───────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_returns_string_not_exception(self, tmp_path: Path) -> None:
        """A timed-out command must return an error string, not raise."""
        output_file = tmp_path / "timeout_result.txt"
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                timeout=1,
            )
        )
        # sleep 5 will time out after 1s
        cmd = f"sleep 5 && echo ok > {output_file}"
        agent = Agent("a", config=self._make_config(cmd), tools=[shell])
        # Must not raise — the tool should return a "timed out" string
        reply = await agent.ask("run it")
        assert await reply.content() == "done"
        # The file should not exist — command was killed
        assert not output_file.exists()

    # ── ignore ────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ignore_blocks_env_file(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=password")
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, ignore=["**/.env"]))

        tool_results: list[str] = []

        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        agent = Agent("a", config=self._make_config("cat .env"), tools=[shell])
        await agent.ask("show me .env", stream=stream)

        assert tool_results, "No tool result received"
        assert "Access denied" in tool_results[0], f"Expected 'Access denied' but got: {tool_results[0]!r}"
        assert "SECRET" not in tool_results[0], "File content leaked despite ignore pattern"

    # ── exit code ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_exit_code_included_on_failure(self, tmp_path: Path) -> None:
        """A failed command must include [exit code: N] in the tool result."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
        agent = Agent("a", config=self._make_config("exit 42"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 42" in tool_results[0], f"Exit code missing: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_exit_code_absent_on_success(self, tmp_path: Path) -> None:
        """A successful command must NOT include [exit code: ...] in the result."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code" not in tool_results[0], f"Unexpected exit code in success: {tool_results[0]!r}"

    # ── filesystem persistence ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_files_persist_between_ask_calls(self, tmp_path: Path) -> None:
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
        agent = Agent(
            "a",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[
                            ToolCallEvent(
                                arguments=json.dumps({"command": "echo 42 > counter.txt"}),
                                name="run_shell_command",
                            )
                        ]
                    )
                ),
                "created",
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[
                            ToolCallEvent(
                                arguments=json.dumps({"command": "cat counter.txt"}),
                                name="run_shell_command",
                            )
                        ]
                    )
                ),
                "read",
            ),
            tools=[shell],
        )

        reply1 = await agent.ask("create counter")
        assert (tmp_path / "counter.txt").exists()

        reply2 = await reply1.ask("read counter")
        assert await reply2.content() == "read"
        assert (tmp_path / "counter.txt").read_text().strip() == "42"

    # ── output truncation ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_output_truncated_when_exceeds_limit(self, tmp_path: Path) -> None:
        """Output longer than max_output must be cut with a note appended."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, max_output=20))
        # Generate 100 chars of output
        agent = Agent("a", config=self._make_config("python3 -c \"print('x' * 100)\""), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        result = tool_results[0]
        assert "truncated" in result, f"Expected truncation note but got: {result!r}"
        # Output was 100 'x' chars; with max_output=20 only 20 should appear
        result = result.replace("TextInput", "")
        assert result.count("x") == 20, f"Expected exactly 20 'x' chars, got {result.count('x')}"

    @pytest.mark.asyncio
    async def test_output_not_truncated_within_limit(self, tmp_path: Path) -> None:
        """Short output must be returned as-is without any truncation note."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, max_output=1000))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "truncated" not in tool_results[0], "Unexpected truncation note for short output"

    # ── timeout exit code 124 ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_returns_exit_code_124(self, tmp_path: Path) -> None:
        """Timed-out commands must include [exit code: 124] (Unix convention)."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, timeout=1))
        agent = Agent("a", config=self._make_config("sleep 5"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 124" in tool_results[0], f"Expected exit code 124 but got: {tool_results[0]!r}"

    # ── readonly ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_readonly_blocks_write_commands(self, tmp_path: Path) -> None:
        """readonly=True must block touch, rm, mkdir."""
        output = tmp_path / "should_not_exist.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, readonly=True))
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was not blocked by readonly=True"

    @pytest.mark.asyncio
    async def test_readonly_allows_read_commands(self, tmp_path: Path) -> None:
        """readonly=True must allow cat, ls, grep."""
        (tmp_path / "hello.txt").write_text("world")

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, readonly=True))
        agent = Agent("a", config=self._make_config("cat hello.txt"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "world" in tool_results[0], f"cat was blocked by readonly=True: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_readonly_overridden_by_explicit_allowed(self, tmp_path: Path) -> None:
        """explicit allowed= takes precedence over readonly=True."""
        output = tmp_path / "out.txt"
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                readonly=True,
                allowed=["touch"],  # user explicitly allows touch despite readonly
            )
        )
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert output.exists(), "touch should be allowed when explicit allowed= overrides readonly"

    # ── workdir in tool description ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_workdir_in_tool_description(self, tmp_path: Path) -> None:
        """The shell tool description must include the working directory path."""
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))

        schemas = await shell.schemas(None)  # type: ignore[arg-type]
        description = schemas[0].function.description
        assert str(tmp_path) in description, f"workdir not in description: {description!r}"


class TestCommandValidationChain:
    """Per-segment whitelist validation: pipelines and chain operators."""

    def test_pipe_passes_when_all_segments_allowed(self) -> None:
        assert (
            validate_command("git log | grep error", allowed=["git", "grep"], blocked=None, forbid_redirects=False)
            is None
        )

    def test_pipe_rejects_when_one_segment_disallowed(self) -> None:
        result = validate_command("echo a | sh", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "sh" in result

    def test_and_chain_passes_when_all_allowed(self) -> None:
        assert validate_command("cat a && cat b", allowed=["cat"], blocked=None, forbid_redirects=False) is None

    def test_or_chain_passes_when_all_allowed(self) -> None:
        assert validate_command("cat a || cat b", allowed=["cat"], blocked=None, forbid_redirects=False) is None

    def test_semicolon_chain_passes_when_all_allowed(self) -> None:
        assert validate_command("echo a; echo b", allowed=["echo"], blocked=None, forbid_redirects=False) is None

    def test_chain_rejects_when_one_segment_disallowed(self) -> None:
        result = validate_command("echo a; rm b", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "rm" in result

    def test_quoted_separator_is_one_segment(self) -> None:
        # `echo "a; b"` is a single command — the `;` is inside an argument.
        assert validate_command('echo "a; b"', allowed=["echo"], blocked=None, forbid_redirects=False) is None

    def test_quoted_pipe_is_one_segment(self) -> None:
        assert validate_command('echo "a | b"', allowed=["echo"], blocked=None, forbid_redirects=False) is None


class TestCommandValidationSubstitution:
    """Substitution and compound constructs are rejected under whitelist."""

    def test_command_substitution_dollar_paren_rejected(self) -> None:
        result = validate_command("echo $(date)", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "command substitution" in result

    def test_command_substitution_backticks_rejected(self) -> None:
        result = validate_command("echo `date`", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "command substitution" in result

    def test_process_substitution_rejected(self) -> None:
        result = validate_command("cat <(echo x)", allowed=["cat"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "process substitution" in result

    def test_compound_if_rejected(self) -> None:
        result = validate_command("if true; then echo x; fi", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None
        assert "compound" in result

    def test_substitution_allowed_without_whitelist(self) -> None:
        # When no allowed list is set, substitution is permitted — restricted
        # mode is what triggers the check.
        assert validate_command("echo $(date)", allowed=None, blocked=None, forbid_redirects=False) is None


class TestCommandValidationRedirect:
    """Redirections are blocked only in pure read-only mode."""

    def test_redirect_allowed_when_explicit_allowed(self) -> None:
        # Explicit allowed=… means the user trusts the whitelist; redirect OK.
        assert validate_command("echo hi > out.txt", allowed=["echo"], blocked=None, forbid_redirects=False) is None

    def test_redirect_rejected_in_readonly(self) -> None:
        result = validate_command("cat f > o.txt", allowed=None, blocked=None, forbid_redirects=True)
        assert result is not None
        assert "redirection" in result

    def test_input_redirect_rejected_in_readonly(self) -> None:
        # `<` is also a redirection — read-only mode forbids any redirection
        # because we cannot statically prove the target isn't a write.
        result = validate_command("cat < in.txt", allowed=None, blocked=None, forbid_redirects=True)
        assert result is not None
        assert "redirection" in result


class TestCommandValidationBlocked:
    """The blocked list applies per-segment, just like allowed."""

    def test_blocked_rejects_segment_inside_chain(self) -> None:
        result = validate_command("echo a; rm b", allowed=None, blocked=["rm"], forbid_redirects=False)
        assert result is not None
        assert "rm" in result

    def test_blocked_does_not_reject_unrelated_chain(self) -> None:
        assert validate_command("echo a; echo b", allowed=None, blocked=["rm"], forbid_redirects=False) is None


class TestCommandValidationParsing:
    """Malformed bash fails closed; permissive mode is unchanged."""

    def test_malformed_bash_fails_closed_under_whitelist(self) -> None:
        result = validate_command("echo $(", allowed=["echo"], blocked=None, forbid_redirects=False)
        assert result is not None

    def test_no_policy_means_permissive(self) -> None:
        # No allowed, no blocked, no readonly — anything goes.
        assert validate_command("rm -rf /", allowed=None, blocked=None, forbid_redirects=False) is None

    def test_empty_command_is_permitted(self) -> None:
        assert validate_command("   ", allowed=["echo"], blocked=None, forbid_redirects=False) is None


class TestReadonlyRedirectIntegration:
    """End-to-end: readonly=True must reject `cat f > out.txt` at run()."""

    def test_readonly_blocks_redirect_via_run(self, tmp_path: Path) -> None:
        (tmp_path / "src.txt").write_text("payload")
        env = LocalShellEnvironment(path=tmp_path, readonly=True)
        result = env.run("cat src.txt > leaked.txt")
        assert "redirection" in result
        assert not (tmp_path / "leaked.txt").exists()

    def test_explicit_allowed_overrides_readonly_for_redirect(self, tmp_path: Path) -> None:
        # When the user pairs readonly=True with an explicit allowed list, the
        # explicit list wins — redirection is permitted, matching the existing
        # `test_readonly_overridden_by_explicit_allowed` semantics.
        env = LocalShellEnvironment(path=tmp_path, readonly=True, allowed=["echo"])
        result = env.run("echo ok > out.txt")
        assert "redirection" not in result
        assert (tmp_path / "out.txt").read_text().strip() == "ok"
