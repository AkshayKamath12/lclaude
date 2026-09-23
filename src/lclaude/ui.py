"""Terminal presentation and I/O handling for lclaude."""

import re
import sys
from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.output import Output
from prompt_toolkit.validation import Validator
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


class StreamAbortedError(Exception):
    """Raised when token streaming is interrupted by the user (Ctrl+C)."""


class _PromptHistory(InMemoryHistory):
    """Store original pasted content while the editor displays compact markers."""

    def __init__(self, pasted_blocks: list[tuple[str, str]]) -> None:
        super().__init__()
        self._pasted_blocks = pasted_blocks

    def append_string(self, string: str) -> None:
        for placeholder, actual in self._pasted_blocks:
            string = string.replace(placeholder, actual, 1)
        super().append_string(string)


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
        interactive = input_stream is not None or output_stream is not None or (
            sys.stdin.isatty() and sys.stdout.isatty()
        )
        self._prompt: PromptSession[str] | None = None
        self._pasted_blocks: list[tuple[str, str]] = []

        if interactive:
            bindings = KeyBindings()

            @bindings.add("enter")
            def accept(event: KeyPressEvent) -> None:
                if self._has_nonblank_content(event.current_buffer.text):
                    event.current_buffer.validate_and_handle()

            @bindings.add("escape", "enter")
            @bindings.add("escape", "c-j")
            def newline(event: KeyPressEvent) -> None:
                event.current_buffer.insert_text("\n")

            # Delete a pasted placeholder as one unit.
            @bindings.add("backspace")
            @bindings.add("c-h")
            def backspace(event: KeyPressEvent) -> None:
                buff = event.current_buffer
                text_before = buff.document.text_before_cursor
                # Check if the cursor is immediately after a pasted placeholder
                match = re.search(r"(<pasted \d+ lines>)$", text_before)
                if match:
                    placeholder = match.group(1)
                    # Delete the full length of the placeholder
                    buff.delete_before_cursor(count=len(placeholder))
                    # Remove it from our background storage to prevent ghost data
                    for i in reversed(range(len(self._pasted_blocks))):
                        if self._pasted_blocks[i][0] == placeholder:
                            self._pasted_blocks.pop(i)
                            break
                else:
                    # Default behavior: delete 1 character
                    buff.delete_before_cursor(count=1)

            if sys.platform == "win32":
                @bindings.add("c-z")
                def windows_eof(event: KeyPressEvent) -> None:
                    if not event.current_buffer.text:
                        event.app.exit(exception=EOFError())

            self._prompt = PromptSession[str](
                "\n> ",
                multiline=True,
                prompt_continuation="... ",
                history=_PromptHistory(self._pasted_blocks),
                enable_history_search=False,
                key_bindings=bindings,
                validator=Validator.from_callable(
                    self._has_nonblank_content
                ),
                validate_while_typing=False,
                input=input_stream,
                output=output_stream,
            )

            # Normalize paste newlines and display large pastes as compact markers.
            original_insert_text = self._prompt.default_buffer.insert_text

            def custom_insert_text(
                data: str,
                overwrite: bool = False,
                move_cursor: bool = True,
                fire_event: bool = True,
            ) -> None:
                data = data.replace("\r\n", "\n").replace("\r", "\n")

                if "\n" in data.strip("\n"):
                    line_count = data.count("\n") + 1
                    placeholder = f"<pasted {line_count} lines>"
                    self._pasted_blocks.append((placeholder, data))
                    original_insert_text(placeholder, overwrite, move_cursor, fire_event)
                else:
                    original_insert_text(data, overwrite, move_cursor, fire_event)

            self._prompt.default_buffer.insert_text = custom_insert_text  # type: ignore[method-assign]

    def _has_nonblank_content(self, text: str) -> bool:
        for placeholder, actual in self._pasted_blocks:
            text = text.replace(placeholder, actual, 1)
        return bool(text.strip())

    def _restore_pasted_text(self, text: str) -> str:
        for placeholder, actual in self._pasted_blocks:
            text = text.replace(placeholder, actual, 1)
        self._pasted_blocks.clear()
        return text

    def read(self) -> str | None:
        """Return a complete prompt, or None after interruption/EOF and cleanup."""
        try:
            if self._prompt is not None:
                user_text = self._prompt.prompt()
                return self._restore_pasted_text(user_text)

            return input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            self._pasted_blocks.clear()
            return None


def render_stream(token_stream: Iterable[str]) -> str:
    """Streams response tokens and renders live Markdown concurrently."""
    accumulated: list[str] = []
    
    # Fallback for piped/redirected output (headless mode)
    if not sys.stdout.isatty():
        sys.stdout.write("Assistant: ")
        try:
            for token in token_stream:
                sys.stdout.write(token)
                sys.stdout.flush()
                accumulated.append(token)
        except KeyboardInterrupt:
            raise StreamAbortedError() from None
        sys.stdout.write("\n")
        return "".join(accumulated)

    # Interactive live terminal rendering
    console = Console(file=sys.stdout, force_terminal=True)
    try:
        with console.status("Thinking…"):
            tokens = iter(token_stream)
            for token in tokens:
                if token:
                    accumulated.append(token)
                    break

        # refresh_per_second=15 limits the repaint rate to prevent terminal flickering
        with Live(
            Markdown(f"**Assistant:**\n\n{''.join(accumulated)}"),
            console=console, 
            refresh_per_second=15,
            vertical_overflow="visible"
        ) as live:
            for token in tokens:
                accumulated.append(token)
                # Live handles the terminal escape diffing automatically
                live.update(Markdown(f"**Assistant:**\n\n{''.join(accumulated)}"))
    except KeyboardInterrupt:
        raise StreamAbortedError() from None
        
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
