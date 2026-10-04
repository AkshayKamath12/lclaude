"""Inline Markdown streaming with append-only history and a bounded live tail."""

import io
import re
import unicodedata
from time import monotonic

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text


class StreamDisplay:
    """Own the response viewport; all cursor movement stays inside that viewport.

    Committed rows are never redrawn. Two trailing rows remain mutable so prose
    wrapping and incomplete Markdown can settle before entering scrollback.
    Tables remain mutable because later rows can change every column width.
    """

    def __init__(self, console: Console, status: str = "") -> None:
        self.console = console
        self.status = status
        self.source = ""
        self.emitted = 0
        self.rows: list[str] = []
        self.width = max(2, console.size.width - 1)
        self.last_frame = 0.0
        self.closed = False

    def _render(self) -> list[str]:
        buffer = io.StringIO()
        renderer = Console(
            file=buffer, width=self.width, force_terminal=True,
            legacy_windows=False,
        )
        renderer.print(Markdown(self.source), end="")
        return buffer.getvalue().splitlines()

    def _frame(self, rows: list[str], commit: list[str]) -> None:
        """Insert history and update changed viewport rows in one write.

        The cursor is always at the start of the last owned row. New rows are
        allocated with linefeeds, allowing the terminal to scroll naturally.
        No scroll margins, absolute screen coordinates, or alternate buffers.
        """
        if rows == self.rows and not commit:
            return
        old = self.rows
        output = ["\x1b[?2026h", "\x1b[?25l", "\r"]
        if len(old) > 1:
            output.append(f"\x1b[{len(old) - 1}A")
        if commit:
            # These rows become immutable history. Existing viewport rows are
            # reused first; additional linefeeds extend native scrollback.
            for row in commit:
                output.extend(("\x1b[2K", row, "\r\n"))
            old = old[len(commit):]
        count = max(len(old), len(rows))
        for index in range(count):
            row = rows[index] if index < len(rows) else ""
            previous = old[index] if index < len(old) else None
            if row != previous:
                output.append(self._changed_row(previous, row))
            if index + 1 < count:
                output.append("\n")
        if count > len(rows):
            output.append(f"\x1b[{count - len(rows)}A")
        output.append("\x1b[?2026l")
        self.console.file.write("".join(output))
        self.console.file.flush()
        self.rows = rows

    def _changed_row(self, previous: str | None, row: str) -> str:
        """Preserve the unchanged styled prefix, including wide Unicode cells."""
        if previous is None:
            return "\x1b[2K" + row + "\r"
        before, after = Text.from_ansi(previous), Text.from_ansi(row)
        prefix = 0
        for index, (old_char, new_char) in enumerate(zip(before.plain, after.plain, strict=False)):
            if old_char != new_char or (
                before.get_style_at_offset(self.console, index)
                != after.get_style_at_offset(self.console, index)
            ):
                break
            prefix += 1
        # Do not split a base character from a combining mark.
        if prefix < len(after) and unicodedata.combining(after.plain[prefix]):
            prefix = max(0, prefix - 1)
        columns = cell_len(after.plain[:prefix])
        buffer = io.StringIO()
        renderer = Console(file=buffer, force_terminal=True, width=self.width)
        renderer.print(after[prefix:], end="", soft_wrap=True)
        movement = f"\x1b[{columns}C" if columns else ""
        return movement + "\x1b[0K" + buffer.getvalue() + "\r"

    def start(self) -> None:
        self.console.print("[bold]Assistant:[/bold]")
        self._frame(["Thinking…", *self._status()], [])

    def _status(self) -> list[str]:
        return [self.status[:self.width]] if self.status else []

    def update(self, token: str, *, force: bool = False) -> None:
        self.source += token
        now = monotonic()
        if not force and now - self.last_frame < 1 / 30:
            return
        self.last_frame = now
        # Freeze wrapping width during a response: terminal-native scrollback
        # cannot be reflowed safely by replaying previously committed Markdown.
        rendered = self._render()
        table = re.search(r"(?m)^\s*\|?.+\|.+\n\s*\|?\s*:?-{3,}", self.source)
        open_table = table is not None and "\n\n" not in self.source[table.start():]
        stable = self.emitted if open_table else max(self.emitted, len(rendered) - 2)
        commit = rendered[self.emitted:stable]
        self.emitted = stable
        available = max(1, min(3, self.console.size.height - len(self._status()) - 1))
        tail = rendered[self.emitted:]
        self._frame((tail[-available:] or [""]) + self._status(), commit)

    def finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            remaining = self._render()[self.emitted:] if self.source else []
            self._frame([""], remaining)
        finally:
            self.console.file.write("\x1b[?2026l\x1b[?25h\r")
            self.console.file.flush()

    def suspend(self) -> None:
        """Release the viewport before tool output or an approval prompt."""
        rendered = self._render() if self.source else []
        self._frame([""], rendered[self.emitted:])
        self.emitted = len(rendered)
        self.rows = []
        self.console.file.write("\x1b[?25h\r")
        self.console.file.flush()
