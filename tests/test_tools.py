"""The single command tool: validation, bounded capture, and subprocess outcomes."""

import io
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from lclaude import ui
from lclaude.tools import (
    OUTPUT_BYTES,
    SHELL,
    TOOL_SCHEMAS,
    ToolError,
    run_command,
    validate_command,
)


def request(tmp_path, command, **kwargs):
    return validate_command(tmp_path, "run_command", {
        "command": command, "shell": SHELL, **kwargs,
    })


def test_tool_definition_explains_execution_and_per_call_directory():
    description = TOOL_SCHEMAS[0]["function"]["description"]
    assert "emit this structured tool call" in description
    assert "a code block or saying you will run it does not run it" in description
    assert "Each call starts a fresh shell" in description
    assert "set cwd on each call" in description


@pytest.mark.parametrize("arguments", [
    None, [], {}, {"command": ""}, {"command": "echo hi", "shell": "bash"},
    {"command": "echo hi", "shell": SHELL, "extra": 1},
    {"command": "echo hi", "shell": SHELL, "timeout_seconds": True},
    {"command": "echo hi", "shell": SHELL, "timeout_seconds": 0},
    {"command": "echo hi", "shell": SHELL, "timeout_seconds": 301},
    {"command": "echo hi", "shell": SHELL, "cwd": 1},
    {"command": "echo hi", "shell": SHELL, "cwd": "missing"},
    {"command": "echo\0hi", "shell": SHELL},
])
def test_argument_validation(tmp_path, arguments):
    with pytest.raises(ToolError):
        validate_command(tmp_path, "run_command", arguments)


def test_unknown_tool_and_non_directory(tmp_path):
    with pytest.raises(ToolError, match="Unknown tool"):
        validate_command(tmp_path, "read_file", {})
    (tmp_path / "file").write_text("hello")
    with pytest.raises(ToolError, match="not a directory"):
        request(tmp_path, "echo hi", cwd="file")


def test_resolves_relative_and_absolute_directories(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    assert request(tmp_path, "echo hi", cwd="child").cwd == child.resolve()
    # Working directory is not containment; absolute directories are allowed.
    assert request(child, "echo hi", cwd=str(tmp_path)).cwd == tmp_path.resolve()


def test_success_stderr_exit_code_and_working_directory(tmp_path):
    command = ("[Console]::Out.Write((Get-Location).Path); "
               "[Console]::Error.Write('problem'); exit 7"
               if SHELL == "powershell" else "pwd; printf problem >&2; exit 7")
    result = run_command(request(tmp_path, command))
    assert result["status"] == "nonzero_exit"
    assert result["exit_code"] == 7
    assert str(tmp_path) in result["stdout"]
    assert result["stderr"] == "problem"
    assert not result["truncated"]


def test_success_and_stdin_is_closed(tmp_path):
    command = ("[Console]::Out.Write('hello'); [Console]::In.ReadToEnd()"
               if SHELL == "powershell" else "printf hello; cat")
    result = run_command(request(tmp_path, command))
    assert result["status"] == "success"
    assert result["stdout"].rstrip("\r\n") == "hello"


def test_both_streams_are_bounded_and_drained(tmp_path):
    command = ("[Console]::Out.Write('x' * 100000); [Console]::Error.Write('y' * 100000)"
               if SHELL == "powershell" else
               "head -c 100000 /dev/zero | tr '\\000' x; "
               "head -c 100000 /dev/zero | tr '\\000' y >&2")
    result = run_command(request(tmp_path, command))
    assert result["status"] == "success"
    assert result["stdout"] == "x" * OUTPUT_BYTES
    assert result["stderr"] == "y" * OUTPUT_BYTES
    assert result["truncated"]
    assert "omitted output was not saved" in result["detail"]


def test_timeout(tmp_path):
    command = "Start-Sleep -Seconds 10" if SHELL == "powershell" else "sleep 10"
    result = run_command(request(tmp_path, command, timeout_seconds=1))
    assert result["status"] == "timeout"


def test_cancellation_stops_spawned_process(tmp_path):
    command = "Start-Sleep -Seconds 10" if SHELL == "powershell" else "sleep 10"
    real_sleep = time.sleep
    interrupted = False

    def interrupt_once(seconds):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    with patch("lclaude.tools.time.sleep", side_effect=interrupt_once):
        result = run_command(request(tmp_path, command))
    assert result["status"] == "cancelled"
    assert result["exit_code"] is not None
    assert "partial effects" in result["detail"]


def test_spawn_error(tmp_path):
    with patch("lclaude.tools.subprocess.Popen", side_effect=OSError("missing shell")):
        result = run_command(request(tmp_path, "echo hi"))
    assert result["status"] == "error"
    assert "missing shell" in result["detail"]


@pytest.mark.parametrize("platform,shell,expected", [
    ("win32", "powershell", "powershell.exe"),
    ("linux", "sh", "/bin/sh"),
    ("darwin", "sh", "/bin/sh"),
])
def test_explicit_shell_invocation(tmp_path, platform, shell, expected):
    from lclaude.tools import Command

    with patch("lclaude.tools.sys.platform", platform), patch.object(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True,
    ), patch(
        "lclaude.tools.subprocess.Popen", side_effect=OSError("inspection")
    ) as spawn:
        run_command(Command("echo hi", shell, tmp_path))
    kwargs = spawn.call_args.kwargs
    argv = spawn.call_args.args[0]
    assert argv[0].endswith(expected)
    assert kwargs["cwd"] == tmp_path
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "shell" not in kwargs


@pytest.mark.parametrize("answer,expected", [("y", "approved"), ("YES", "approved"),
                                            ("", "rejected"), ("no", "rejected")])
def test_approval_shows_entire_request_before_prompt(tmp_path, answer, expected, capsys):
    command = request(tmp_path, "echo " + "x" * 1000)
    def approve(prompt):
        visible = capsys.readouterr().out
        assert command.command in visible
        assert str(tmp_path) in visible
        assert command.shell in visible and "not a sandbox" in visible
        return answer
    with patch.object(sys.stdin, "isatty", return_value=True), patch(
        "builtins.input", side_effect=approve
    ):
        assert ui.approve_command(command) == expected


@pytest.mark.parametrize("failure,expected", [(EOFError(), "rejected"),
                                             (KeyboardInterrupt(), "cancelled")])
def test_approval_interruption(tmp_path, failure, expected):
    with patch.object(sys.stdin, "isatty", return_value=True), patch(
        "builtins.input", side_effect=failure
    ):
        assert ui.approve_command(request(tmp_path, "echo hi")) == expected


def test_noninteractive_never_approves(tmp_path):
    with patch("sys.stdin", io.StringIO("yes\n")), patch("builtins.input") as prompt:
        assert ui.approve_command(request(tmp_path, "echo hi")) == "rejected"
    prompt.assert_not_called()


def test_terminal_controls_are_escaped(tmp_path, capsys):
    with patch.object(sys.stdin, "isatty", return_value=False):
        ui.approve_command(request(tmp_path, "echo \x1b[2J"))
    assert "\x1b" not in capsys.readouterr().out


def test_unsupported_shell_is_never_substituted(tmp_path):
    from lclaude.tools import Command

    with patch("lclaude.tools.subprocess.Popen") as spawn:
        result = run_command(Command("echo hi", "bash", tmp_path))
    spawn.assert_not_called()
    assert result["status"] == "error"


def test_cleanup_failure_is_reported():
    from unittest.mock import MagicMock

    from lclaude.tools import _stop

    process = MagicMock()
    with patch("lclaude.tools.sys.platform", "win32"), patch(
        "lclaude.tools.subprocess.run", side_effect=OSError("cleanup unavailable"),
    ):
        assert "could not be confirmed" in _stop(process)
