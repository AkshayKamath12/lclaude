"""Terminal presentation and I/O handling for lclaude."""

from collections.abc import Iterable
import sys
import signal


class StreamAbortedError(Exception):
    """Raised when token streaming is interrupted by the user (Ctrl+C)."""

def print_banner(model: str, host: str) -> None:
    sys.stdout.write("==================================================\n")
    sys.stdout.write("  lclaude - Local CLI Coding Assistant\n")
    sys.stdout.write(f"  Model:   {model}\n")
    sys.stdout.write(f"  Host:    {host}\n")
    sys.stdout.write("  Commands: /clear, /history, /exit, /help\n")
    sys.stdout.write("  Abort generation: Ctrl+C | Exit: Ctrl+C at prompt\n")
    sys.stdout.write("==================================================\n")

def get_user_input() -> str | None:
    """Prompts the user for input.

    Returns:
        The input string stripped, or None if interrupted or EOF received.
    """
    try:
        return input("\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

def render_stream(token_stream: Iterable[str]) -> str:
    """Streams response tokens directly to stdout in real-time.

    Returns:
        The fully accumulated response string.

    Raises:
        StreamAbortedError: If KeyboardInterrupt is received mid-stream.
    """
    sys.stdout.write("\nAssistant: ")
    sys.stdout.flush()

    accumulated: list[str] = []
    try:
        for token in token_stream:
            sys.stdout.write(token)
            sys.stdout.flush()
            accumulated.append(token)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise StreamAbortedError()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(accumulated)


def print_aborted() -> None:
    """Prints generation abort confirmation."""
    sys.stdout.write("\n[Generation aborted by user]\n")
    sys.stdout.flush()


def print_session_end() -> None:
    """Prints exit notification."""
    sys.stdout.write("\n\nSession terminated by user.\n")
    sys.stdout.flush()


def print_error(title: str, message: str) -> None:
    """Prints formatted error messages to stderr."""
    sys.stderr.write(f"\n[{title}]: {message}\n")
    sys.stderr.flush()


def print_startup_error(message: str) -> None:
    """Prints fatal startup verification failures to stderr."""
    sys.stderr.write(f"Startup check failed: {message}\n")
    sys.stderr.flush()