"""One command tool: validation and subprocesses, independent of UI and sessions."""

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

SHELL = "powershell" if sys.platform == "win32" else "sh"
OUTPUT_BYTES = 6000  # Per stream; continue draining after reaching the capture limit.
TOOL_SCHEMAS: list[dict[str, Any]] = [{
    "type": "function", "function": {
        "name": "run_command",
        "description": (
            f"Run a command using {SHELL}, after user approval. "
            "When the user asks you to run a command, emit this structured tool call "
            "in the same response; a code block or saying you will run it does not run it. "
            "Only report execution after reading a tool result. Each call starts a fresh "
            "shell, so changing directories does not persist; set cwd on each call. "
            "Use syntax for the specified shell. The working directory is not a sandbox. "
            "Output is bounded and may be truncated."
        ),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "command": {"type": "string"},
                "shell": {"type": "string", "enum": [SHELL]},
                "cwd": {"type": "string", "description": "Relative to project root; default ."},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
            },
            "required": ["command", "shell"],
        },
    },
}]


class ToolError(ValueError):
    """The model supplied an invalid command request."""


@dataclass(frozen=True)
class Command:
    command: str
    shell: str
    cwd: Path
    timeout_seconds: int = 60


def validate_command(project: Path, name: str, arguments: Any) -> Command:
    if name != "run_command":
        raise ToolError(f"Unknown tool: {name}")
    if not isinstance(arguments, dict) or set(arguments) - {
        "command", "shell", "cwd", "timeout_seconds",
    }:
        raise ToolError("Expected command, shell, optional cwd and timeout_seconds")
    command, shell = arguments.get("command"), arguments.get("shell")
    if not isinstance(command, str) or not command.strip() or "\0" in command:
        raise ToolError("command must be a nonempty string without NUL characters")
    if sys.platform not in ("win32", "linux", "darwin") or shell != SHELL:
        raise ToolError(f"This platform requires shell={SHELL!r}; no shell substitution is allowed")
    cwd = arguments.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd or "\0" in cwd:
        raise ToolError("cwd must be a nonempty directory path")
    timeout = arguments.get("timeout_seconds", 60)
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ToolError("timeout_seconds must be an integer from 1 to 300")
    try:
        directory = (project / cwd).resolve(strict=True)
        if not directory.is_dir():
            raise ToolError(f"Working directory is not a directory: {directory}")
    except (OSError, RuntimeError) as exc:
        raise ToolError(f"Cannot use working directory: {exc}") from exc
    return Command(command, shell, directory, timeout)


def command_result(status: str, detail: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **fields}


def _stop(process: subprocess.Popen[bytes]) -> str:
    """Stop the process group/tree; report cleanup failure."""
    try:
        if sys.platform == "win32":
            stopped = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
            if stopped.returncode:
                if process.poll() is None:
                    process.kill()
                return "Process tree cleanup could not be confirmed; descendants may remain."
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Process cleanup could not be confirmed: {exc}"
    return ""


def run_command(command: Command) -> dict[str, Any]:
    """Execute an approved request with bounded capture and process tree cancellation.

    Deliberately detached processes can escape cleanup; this is not containment.
    """
    if command.shell == "powershell":
        executable = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                         "System32/WindowsPowerShell/v1.0/powershell.exe")
        argv = [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); " + command.command]
    elif command.shell == "sh":
        argv = ["/bin/sh", "-c", command.command]
    else:
        return command_result("error", f"Unsupported shell: {command.shell}")
    try:
        process = subprocess.Popen(
            argv, cwd=command.cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=sys.platform != "win32",
            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                           if sys.platform == "win32" else 0),
        )
    except OSError as exc:
        return command_result("error", f"Could not start command: {exc}")

    buffers = [bytearray(), bytearray()]
    truncated = [False, False]
    errors: list[str] = []

    def drain(stream: BinaryIO, index: int) -> None:
        try:
            with stream:
                while block := stream.read(4096):
                    remaining = OUTPUT_BYTES - len(buffers[index])
                    buffers[index].extend(block[:remaining])
                    truncated[index] |= len(block) > remaining
        except OSError as exc:
            errors.append(str(exc))

    assert process.stdout is not None and process.stderr is not None
    threads = [threading.Thread(target=drain, args=(stream, index), daemon=True)
               for index, stream in enumerate((process.stdout, process.stderr))]
    status, detail = "success", "Command completed."
    deadline = time.monotonic() + command.timeout_seconds
    try:
        for thread in threads:
            thread.start()
        while process.poll() is None or any(t.is_alive() for t in threads):
            if time.monotonic() >= deadline:
                status, detail = "timeout", "Command timed out; partial effects may have occurred."
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        status, detail = "cancelled", "Command cancelled; partial effects may have occurred."
    finally:
        if status != "success" or process.poll() is None:
            detail += " " + _stop(process)
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=0.5)
    if any(t.is_alive() for t in threads):
        detail += " Output pipes remain open; detached processes may still be running."
        truncated = [True, True]
    if status == "success" and process.returncode != 0:
        status, detail = "nonzero_exit", "Command returned a nonzero exit code."
    if errors:
        if status in ("success", "nonzero_exit"):
            status = "error"
        detail += " Output capture failed: " + "; ".join(errors)
    if any(truncated):
        detail += " Output truncated; omitted output was not saved."
    return command_result(
        status, detail.strip(), exit_code=process.returncode,
        stdout=bytes(buffers[0]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers[1]).decode("utf-8", errors="replace"),
        truncated=any(truncated),
    )
