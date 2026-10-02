"""Terminal presentation and I/O handling for lclaude."""

import json
import re
import sys
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, ConditionalCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from lclaude.context import ContextBudget, PromptCount
from lclaude.engine import Usage
from lclaude.tools import Command


class StreamAbortedError(Exception):
    """Raised when token streaming is interrupted by the user (Ctrl+C)."""


def choose_model(
    models: list[str], current: str, *,
    input_stream: Input | None = None, output_stream: Output | None = None,
) -> str | None:
    """Return a selection, or None; never query or mutate the inference engine."""
    if not models:
        print_error("Model selection", "No models available. Download one with ollama pull.")
        return None
    interactive = input_stream is not None or output_stream is not None or (
        sys.stdin.isatty() and sys.stdout.isatty()
    )
    if not interactive:
        sys.stdout.write("\nAvailable models (select with /model <name>):\n")
        for name in models:
            sys.stdout.write(f"  {name}{' (current)' if name == current else ''}\n")
        return None

    selected = models.index(current) if current in models else 0
    bindings = KeyBindings()

    @bindings.add("up")
    def previous(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = (selected - 1) % len(models)

    @bindings.add("down")
    def next_model(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = (selected + 1) % len(models)

    @bindings.add("enter")
    def accept(event: KeyPressEvent) -> None:
        event.app.exit(result=models[selected])

    @bindings.add("escape")
    @bindings.add("c-c")
    @bindings.add("c-d")
    @bindings.add("c-z")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def content() -> FormattedText:
        lines = [("", "Select model | Up/Down: move | Enter: select | Esc: cancel\n")]
        for index, name in enumerate(models):
            label = f"{'>' if index == selected else ' '} {name}"
            if name == current:
                label += " (current)"
            lines.append(("reverse bold" if index == selected else "", label + "\n"))
        return FormattedText(lines)

    control = FormattedTextControl(
        content, focusable=True, get_cursor_position=lambda: Point(x=0, y=selected + 1),
    )
    application: Application[str | None] = Application(
        layout=Layout(Window(control, always_hide_cursor=True)),
        key_bindings=bindings, input=input_stream, output=output_stream,
        full_screen=False, erase_when_done=True,
    )
    try:
        return application.run()
    except (KeyboardInterrupt, EOFError):
        return None


def format_chat_date(timestamp: str) -> str:
    """Display a saved timestamp in local time, without seconds or leading zeroes."""
    local = datetime.fromisoformat(timestamp).astimezone()
    hour = local.hour % 12 or 12
    period = "AM" if local.hour < 12 else "PM"
    return f"{local:%b} {local.day}, {local.year} · {hour}:{local.minute:02d} {period}"


def choose_chat(
    chats: list[tuple[str, str]], current: str, *,
    input_stream: Input | None = None, output_stream: Output | None = None,
) -> str | None:
    """Browse chats in supplied recency order with a bounded, scrollable viewport."""
    if not chats:
        sys.stdout.write("\nNo saved chats for this project.\n")
        return None
    interactive = input_stream is not None or output_stream is not None or (
        sys.stdin.isatty() and sys.stdout.isatty()
    )
    if not interactive:
        sys.stdout.write("\nSaved chats (open /chat in an interactive terminal to resume):\n")
        for chat_id, label in chats:
            sys.stdout.write(f"  {label}{' (current)' if chat_id == current else ''}\n")
        return None

    selected = 0
    bindings = KeyBindings()

    @bindings.add("up")
    def previous(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = max(0, selected - 1)

    @bindings.add("down")
    def next_chat(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = min(len(chats) - 1, selected + 1)

    @bindings.add("pageup")
    def previous_page(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = max(0, selected - 10)

    @bindings.add("pagedown")
    def next_page(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = min(len(chats) - 1, selected + 10)

    @bindings.add("enter")
    def accept(event: KeyPressEvent) -> None:
        event.app.exit(result=chats[selected][0])

    @bindings.add("escape")
    @bindings.add("c-c")
    @bindings.add("c-d")
    @bindings.add("c-z")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def content() -> FormattedText:
        lines = []
        for index, (chat_id, label) in enumerate(chats):
            # Keep each chat on one row, including user-provided preview text.
            label = "".join(char if char.isprintable() else " " for char in label)
            line = f"{'>' if index == selected else ' '} {label}"
            if chat_id == current:
                line += " (current)"
            lines.append(("reverse bold" if index == selected else "", line + "\n"))
        return FormattedText(lines)

    control = FormattedTextControl(
        content, focusable=True, get_cursor_position=lambda: Point(x=0, y=selected),
    )
    header = Window(FormattedTextControl(
        "↑/↓, PgUp/PgDn: scroll | Esc: cancel"
    ), height=1)
    application: Application[str | None] = Application(
        layout=Layout(HSplit([
            header, Window(control, height=min(10, len(chats)), always_hide_cursor=True),
        ]), focused_element=control),
        key_bindings=bindings, input=input_stream, output=output_stream,
        full_screen=False, erase_when_done=True,
    )
    try:
        return application.run()
    except (KeyboardInterrupt, EOFError):
        return None


def print_model_changed(model: str) -> None:
    sys.stdout.write(f"\nModel selected: {model}\n")


def print_model_selection_cancelled() -> None:
    sys.stdout.write("\nModel selection cancelled.\n")


class SlashCommandCompleter(Completer):
    """Present injected command metadata without knowing how commands execute."""

    def __init__(self, commands: Mapping[str, str], models: Iterable[str] = ()) -> None:
        self.commands = commands
        self.models = tuple(models)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text
        if document.cursor_position != len(text):
            return
        model_argument = re.fullmatch(r"/model[ \t]+([^\s]*)", text, re.IGNORECASE)
        if model_argument and "/model" in self.commands:
            prefix = model_argument.group(1)
            for model in self.models:
                if model.lower().startswith(prefix.lower()):
                    yield Completion(model, start_position=-len(prefix))
            return
        if (
            not text.startswith("/")
            or any(char.isspace() for char in text)
            or document.cursor_position != len(text)
        ):
            return
        for name, description in self.commands.items():
            if name.startswith(text.lower()):
                yield Completion(name, start_position=-len(text), display_meta=description)


class _PromptHistory(InMemoryHistory):
    """Store original pasted content while the editor displays compact markers."""

    def __init__(self, pasted_blocks: list[tuple[str, str]]) -> None:
        super().__init__()
        self._pasted_blocks = pasted_blocks

    def append_string(self, string: str) -> None:
        for placeholder, actual in self._pasted_blocks:
            string = string.replace(placeholder, actual, 1)
        super().append_string(string)


def print_banner(model: str, host: str, commands: Iterable[str] = ()) -> None:
    sys.stdout.write("==================================================\n")
    sys.stdout.write("  lclaude - Local CLI Coding Assistant\n")
    sys.stdout.write(f"  Model:   {model}\n")
    sys.stdout.write(f"  Host:    {host}\n")
    sys.stdout.write(f"  Commands: {', '.join(commands)}\n")
    sys.stdout.write("  Submit: Enter | Newline: Alt+Enter or Escape then Enter\n")
    sys.stdout.write("  Abort generation: Ctrl+C | Exit: Ctrl+C at prompt\n")
    sys.stdout.write("==================================================\n")


class InputReader:
    """Own input editing and process-local recall history, separate from chat state."""

    def __init__(
        self, *, commands: Mapping[str, str] | None = None,
        models: Iterable[str] = (),
        input_stream: Input | None = None, output_stream: Output | None = None
    ) -> None:
        interactive = input_stream is not None or output_stream is not None or (
            sys.stdin.isatty() and sys.stdout.isatty()
        )
        self._prompt: PromptSession[str] | None = None
        self._pasted_blocks: list[tuple[str, str]] = []
        self._completion_suppressed = False
        self._context_text = ""

        if interactive:
            bindings = KeyBindings()

            def refresh_completions(buffer: Buffer) -> None:
                # Registry lookup is immediate. Highlight without replacing the
                # typed prefix, so further typing continues to filter normally.
                completions = list(buffer.completer.get_completions(
                    buffer.document, CompleteEvent(text_inserted=True)
                )) if buffer.completer else []
                buffer.complete_state = (
                    CompletionState(buffer.document, completions, complete_index=0)
                    if completions else None
                )

            @bindings.add("tab", filter=has_completions)
            def complete(event: KeyPressEvent) -> None:
                buffer = event.current_buffer
                state = buffer.complete_state
                if state and state.current_completion:
                    buffer.apply_completion(state.current_completion)
                    buffer.complete_state = None

            @bindings.add("escape", filter=has_completions)
            def dismiss_completion(event: KeyPressEvent) -> None:
                event.current_buffer.cancel_completion()

            @bindings.add(Keys.BracketedPaste)
            def bracketed_paste(event: KeyPressEvent) -> None:
                self._completion_suppressed = True
                event.current_buffer.complete_state = None
                event.current_buffer.insert_text(event.data)

            @bindings.add("enter")
            def accept(event: KeyPressEvent) -> None:
                complete(event)
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
                if not buff.text:
                    self._completion_suppressed = False
                if not self._completion_suppressed:
                    refresh_completions(buff)

            if sys.platform == "win32":
                @bindings.add("c-z")
                def windows_eof(event: KeyPressEvent) -> None:
                    if not event.current_buffer.text:
                        event.app.exit(exception=EOFError())

            self._prompt = PromptSession[str](
                "\n> ",
                multiline=True,
                bottom_toolbar=None,
                prompt_continuation="... ",
                history=_PromptHistory(self._pasted_blocks),
                enable_history_search=False,
                completer=ConditionalCompleter(
                    SlashCommandCompleter(commands or {}, models),
                    Condition(lambda: not self._completion_suppressed),
                ),
                complete_while_typing=False,
                style=Style.from_dict({
                    "completion-menu": "bg:default fg:default",
                    "completion-menu.completion": "bg:default fg:default",
                    "completion-menu.meta.completion": "bg:default fg:default",
                    "completion-menu.completion.current": "bg:default fg:default reverse bold",
                    "completion-menu.meta.completion.current": "bg:default fg:default reverse bold",
                }),
                key_bindings=bindings,
                validator=Validator.from_callable(
                    self._has_nonblank_content
                ),
                validate_while_typing=False,
                input=input_stream,
                output=output_stream,
            )
            self._prompt.default_buffer.on_text_insert += refresh_completions

            # Anchor at the slash, not at the cursor that moves while filtering.
            for control in self._prompt.layout.find_all_controls():
                if (
                    isinstance(control, BufferControl)
                    and control.buffer is self._prompt.default_buffer
                ):
                    control.menu_position = lambda: 0

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

    @property
    def context_text(self) -> str:
        return self._context_text

    def set_context(self, count: PromptCount, limit: int | None, budget: ContextBudget) -> None:
        """Replace the footer in place without writing into the transcript."""
        self._context_text = context_status(count, limit, budget)
        if self._prompt is not None:
            # An empty callback result still paints the toolbar background.
            # PromptSession hides the entire row only when the option is None.
            self._prompt.bottom_toolbar = self._context_text or None
            self._prompt.app.invalidate()

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
                self._completion_suppressed = False
                user_text = self._prompt.prompt()
                return self._restore_pasted_text(user_text)

            return input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            self._pasted_blocks.clear()
            return None


def render_conversation(messages: Iterable[Mapping[str, Any]], *, chat_id: str) -> None:
    """Replay saved dialogue into terminal scrollback without running inference."""
    interactive = sys.stdout.isatty()
    console = Console(file=sys.stdout) if interactive else None
    if console is not None:
        console.rule(f"Resumed chat {chat_id}")
    else:
        sys.stdout.write(f"\nResumed chat {chat_id}\n\n")
    calls: list[dict[str, Any]] = []
    try:
        for message in messages:
            role = message["role"]
            if role == "assistant" and message.get("tool_calls"):
                calls = list(message["tool_calls"])
                if not message["content"]:
                    continue
            if role == "tool":
                try:
                    result = json.loads(message["content"])
                except ValueError:
                    result = {}
                truncated = isinstance(result, dict) and bool(result.get("truncated"))
                call = calls.pop(0) if calls else {}
                print_tool_activity(message["tool_name"],
                                    call.get("function", {}).get("arguments", {}),
                                    result.get("status", "unknown"), truncated)
                continue
            if role not in ("user", "assistant"):
                continue
            label = "You" if role == "user" else "Assistant"
            content = message["content"]
            if console is None:
                sys.stdout.write(f"{label}:\n{content}\n\n")
            elif role == "user":
                console.print("You:", style="bold")
                console.print(content, markup=False, highlight=False)
                console.print()
            else:
                console.print(Markdown(f"**Assistant:**\n\n{content}"))
                console.print()
    except KeyboardInterrupt:
        sys.stdout.write("\nTranscript display interrupted. Chat is still active.\n")
    sys.stdout.flush()


def render_stream(token_stream: Iterable[str], *, context_text: str = "") -> str:
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
        with pinned_footer(console, context_text) as refresh_footer:
            return _render_terminal_stream(token_stream, console, refresh_footer)
    except KeyboardInterrupt:
        raise StreamAbortedError() from None


def _render_terminal_stream(
    token_stream: Iterable[str], console: Console, refresh_footer: Callable[[], None],
) -> str:
    accumulated: list[str] = []
    try:
        with console.status("Thinking…"):
            tokens = iter(token_stream)
            for token in tokens:
                refresh_footer()
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
                refresh_footer()
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


def _safe_tool_text(value: Any) -> str:
    """Make control characters visible so a command cannot hide its approval display."""
    return "".join(
        char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
        for char in str(value)
    )


def approve_command(command: Command) -> str:
    """Show the complete request and require explicit interactive approval."""
    sys.stdout.write(
        f"\nrun_command\n  Shell: {command.shell}\n"
        f"  Directory: {_safe_tool_text(command.cwd)}\n"
        f"  Timeout: {command.timeout_seconds}s\n"
        f"  Command: {_safe_tool_text(command.command)}\n"
        "  Runs with your permissions; the directory is not a sandbox.\n"
    )
    sys.stdout.flush()
    if not sys.stdin.isatty():
        sys.stdout.write("Command rejected: approval requires interactive input.\n")
        return "rejected"
    try:
        return "approved" if input("Run this command? [y/N] ").strip().lower() in (
            "y", "yes",
        ) else "rejected"
    except EOFError:
        return "rejected"
    except KeyboardInterrupt:
        return "cancelled"


def print_tool_activity(name: str, arguments: Any, status: str, truncated: bool = False) -> None:
    details = arguments.get("command", "") if isinstance(arguments, dict) else "invalid arguments"
    text = _safe_tool_text(details)
    if len(text) > 120:
        text = text[:117] + "..."
    line = f"  {_safe_tool_text(name)}: {text} - {status}"
    if truncated:
        line += "; output truncated (omitted output not saved)"
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def print_tool_resume() -> None:
    """Explain how interrupted tool work is represented when resuming a chat."""
    sys.stdout.write(
        "\nThe previous turn stopped after tool work. Completed results are saved. "
        "Pending commands have an unknown outcome; not_started commands were not executed. "
        "No commands are repeated automatically.\n"
    )


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


def context_status(count: PromptCount, limit: int | None, budget: ContextBudget) -> str:
    """Show the current estimate and warn when it exceeds the context allocation."""
    if limit is None:
        return ""
    marker = "~" if count.estimated else ""
    status = (
        f"Prompt {marker}{count.total:,} / {limit:,} tokens | "
        f"{budget.response_tokens:,} reserved for reply"
    )
    overflow = count.total + budget.response_tokens - limit
    if overflow > 0:
        status += f" | WARNING: ~{overflow:,} tokens over context budget"
    return status


@contextmanager
def pinned_footer(console: Console, text: str) -> Generator[Callable[[], None], None, None]:
    """Reserve the terminal's bottom row while Rich renders above it.

    PromptSession owns the bottom toolbar during input. During generation, a VT
    scroll region keeps output above the same row, without taking over scrollback
    or switching to an alternate screen. Always restore the region on exit.
    """
    if not text or console.legacy_windows:
        yield lambda: None
        return
    dimensions: tuple[int, int] | None = None

    def refresh() -> None:
        nonlocal dimensions
        width, height = console.size
        if height < 3 or width < 2:
            return
        size = (width, height)
        if size == dimensions:
            return
        dimensions = size
        # Hold Rich's output lock so its animation cannot interleave control bytes.
        with console:
            # Make a blank row below the cursor, including when input ended on
            # the last screen row. Keep Rich's cursor above the reserved footer.
            console.file.write("\n\x1b[1A\r")
            console.file.write(
                f"\x1b7\x1b[1;{height - 1}r\x1b[{height};1H\x1b[2K"
                f"\x1b[7m{text[:width - 1]}\x1b[0m\x1b8"
            )
            console.file.flush()

    try:
        refresh()
        yield refresh
    finally:
        if dimensions is not None:
            with console:
                console.file.write("\x1b7\x1b[r" + f"\x1b[{dimensions[1]};1H\x1b[2K\x1b8")
                console.file.flush()


def print_context(
    count: PromptCount, limit: int | None, budget: ContextBudget, *, detail: bool = False,
) -> None:
    """Explicit /context details; routine updates belong only in the footer."""
    if not detail:
        status = context_status(count, limit, budget)
        if status:
            sys.stdout.write(status + "\n")
        return

    marker = "~" if count.estimated else ""
    capacity = f" / {limit:,}" if limit is not None else ""
    sys.stdout.write(
        "Current context (tokens)\n"
        f"  Instructions   {marker}{count.instructions:,}\n"
        f"  Conversation   {marker}{count.conversation:,}\n"
        f"  Formatting     {marker}{count.overhead:,}\n"
        + (f"  Tools          {marker}{count.tools:,}\n" if count.tools else "")
        +
        f"  Total          {marker}{count.total:,}{capacity}\n"
        f"  Reply limit    {budget.response_tokens:,}\n"
    )


def print_usage(usage: Usage, predicted: PromptCount) -> None:
    """Keep actual usage for the previous request separate from current context."""
    if usage.prompt_eval_count is None and usage.eval_count is None:
        return
    sys.stdout.write("\nLast request (tokens reported by Ollama)\n")
    if usage.prompt_eval_count is not None:
        marker = "~" if predicted.estimated else ""
        label = "estimated" if predicted.estimated else "counted"
        sys.stdout.write(
            f"  Prompt         {usage.prompt_eval_count:,} "
            f"({label} {marker}{predicted.total:,} before sending)\n"
        )
    if usage.eval_count is not None:
        sys.stdout.write(f"  Reply          {usage.eval_count:,}\n")
