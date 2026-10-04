"""Exercise actual terminal contents, including native scrollback."""

import io
from unittest.mock import patch

import pyte
import pytest
from rich.console import Console

from lclaude.stream_display import StreamDisplay
from lclaude.ui import StreamAbortedError, render_stream


class Terminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.screen = pyte.HistoryScreen(60, 8, history=10000)
        self.stream = pyte.Stream(self.screen)
        self.screen.cursor_position(line=7, column=1)

    def isatty(self):
        return True

    def write(self, value):
        self.stream.feed(value)
        return super().write(value)

    def transcript(self):
        history = ["".join(row[k].data for k in sorted(row))
                   for row in self.screen.history.top]
        return "\n".join(history + self.screen.display)


def display():
    terminal = Terminal()
    console = Console(file=terminal, width=60, height=8, force_terminal=True)
    renderer = StreamDisplay(console, "Context status")
    renderer.start()
    return terminal, renderer


@pytest.mark.parametrize("code", [False, True])
def test_long_response_enters_history_once_and_follows_bottom(code):
    terminal, renderer = display()
    if code:
        renderer.update("```python\n", force=True)
    for index in range(200):
        renderer.update(f"row{index:03d}\n" + ("" if code else "\n"), force=True)
        assert f"row{index:03d}" in "\n".join(terminal.screen.display)
    if code:
        renderer.update("```\n", force=True)
    before = terminal.getvalue()
    for _ in range(10):
        renderer.update("", force=True)
    assert terminal.getvalue() == before
    renderer.finish()
    transcript = terminal.transcript()
    assert transcript.count("Assistant:") == 1
    for index in range(200):
        assert transcript.count(f"row{index:03d}") == 1
    assert "Context status" not in transcript
    assert "row199" in "\n".join(terminal.screen.display)


def test_partial_prose_and_markdown_visible_before_completion():
    terminal, renderer = display()
    renderer.update("hello **world**", force=True)
    assert "hello world" in terminal.transcript()
    assert "**world**" not in terminal.transcript()
    assert "\x1b[1m" in terminal.getvalue()
    renderer.update(" next", force=True)
    assert "world next" in terminal.transcript()
    renderer.finish()


@pytest.mark.parametrize("ending", [KeyboardInterrupt, RuntimeError])
def test_stream_failure_restores_terminal_and_preserves_partial_text(ending):
    terminal = Terminal()
    console = Console(file=terminal, width=60, height=8, force_terminal=True)

    def tokens():
        yield "partial reply"
        raise ending()

    with patch("sys.stdout", terminal), patch("lclaude.ui.Console", return_value=console):
        with pytest.raises(StreamAbortedError if ending is KeyboardInterrupt else ending):
            render_stream(tokens(), context_text="Context status")
    assert terminal.transcript().count("partial reply") == 1
    assert "Context status" not in terminal.transcript()
    assert terminal.getvalue().endswith("\x1b[?2026l\x1b[?25h\r")
