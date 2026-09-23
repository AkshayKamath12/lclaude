"""Terminal presentation and I/O handling for lclaude."""

import sys
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import Output
from prompt_toolkit.validation import Validator


class StreamAbortedError(Exception):
    """Raised when token streaming is interrupted by the user (Ctrl+C)."""

def print_banner(model: str, host: str) -> None:
    sys.stdout.write("==================================================\n")
    sys.stdout.write("  lclaude - Local CLI Coding Assistant\n")
    sys.stdout.write(f"  Model:   {model}\n")
    sys.stdout.write(f"  Host:    {host}\n")
    sys.stdout.write("  Commands: /clear, /history, /exit, /help\n")
    sys.stdout.write("  Submit: Enter | Newline: Alt+Enter or Escape then Enter\n")
    sys.stdout.write("  Abort generation: Ctrl+C | Exit: Ctrl+C at prompt\n")
    sys.stdout.write("==================================================\n")

class InputReader:
    """Own input editing and process-local recall history, separate from chat state."""

    def __init__(
        self, *, input_stream: Input | None = None, output_stream: Output | None = None
    ) -> None:
        # Explicit streams support isolated editor tests without a real terminal.
        interactive = input_stream is not None or output_stream is not None or (
            sys.stdin.isatty() and sys.stdout.isatty()
        )
        self._prompt: PromptSession[str] | None = None
        if interactive:
            bindings = KeyBindings()

            @bindings.add("enter")
            def accept(event: KeyPressEvent) -> None:
                if event.current_buffer.text.strip():
                    event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            @bindings.add("escape", "c-j")
            def newline(event: KeyPressEvent) -> None:
                event.current_buffer.insert_text("\n")

            if sys.platform == "win32":
                @bindings.add("c-z")
                def windows_eof(event: KeyPressEvent) -> None:
                    if not event.current_buffer.text:
                        event.app.exit(exception=EOFError())

            self._prompt = PromptSession[str](
                "\n> ",
                multiline=True,
                prompt_continuation="... ",
                history=InMemoryHistory(),
                enable_history_search=False,
                key_bindings=bindings,
                # Also guard alternative library acceptance commands.
                validator=Validator.from_callable(lambda text: bool(text.strip())),
                validate_while_typing=False,
                input=input_stream,
                output=output_stream,
            )

    def read(self) -> str | None:
        """Return a complete prompt, or None after interruption/EOF and cleanup."""
        try:
            if self._prompt is not None:
                return self._prompt.prompt()
            # Preserve the existing line-oriented contract for redirected I/O.
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
        raise StreamAbortedError() from None

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
