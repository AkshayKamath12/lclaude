"""Real keyboard processing for the scrollable chat picker."""

from unittest.mock import patch

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lclaude.commands import SelectChat, handle_slash_command
from lclaude.session import Session
from lclaude.ui import choose_chat


@pytest.mark.parametrize("keys, expected", [
    ("\r", "0"),
    ("\x1b[B\r", "1"),
    ("\x1b[A\r", "0"),
    ("\x1b[B" * 24 + "\r", "24"),
    ("\x1b[6~\r", "10"),
    ("\x1b[6~\x1b[5~\r", "0"),
    ("\x1b[6~" * 4 + "\r", "24"),
    ("\x1b", None), ("\x03", None), ("\x04", None), ("\x1a", None),
])
def test_chat_navigation_and_cancel(keys, expected):
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        assert choose_chat(
            [(str(index), f"Chat {index}") for index in range(25)], "3",
            input_stream=pipe, output_stream=DummyOutput(),
        ) == expected


def test_chat_command_has_no_arguments():
    assert handle_slash_command(" /CHAT ", Session()) == SelectChat()
    assert handle_slash_command("/chat some-id", Session()) is True


def test_empty_chat_picker_never_opens_terminal(capsys):
    with patch("lclaude.ui.Application", side_effect=AssertionError("no terminal")):
        assert choose_chat([], "current") is None
    assert "No saved chats" in capsys.readouterr().out


def test_redirected_chat_picker_lists_without_consuming_input(capsys):
    with (
        patch("sys.stdin.isatty", return_value=False),
        patch("builtins.input", side_effect=AssertionError("no input")),
    ):
        assert choose_chat([("a", "Newest"), ("b", "Older")], "b") is None
    output = capsys.readouterr().out
    assert output.index("Newest") < output.index("Older")
    assert "Older (current)" in output


@pytest.mark.parametrize("utc_time, expected", [
    ("2026-10-01T00:05:00+00:00", "Sep 30, 2026 · 5:05 PM"),
    ("2026-09-30T07:00:00+00:00", "Sep 30, 2026 · 12:00 AM"),
    ("2026-09-30T19:00:00+00:00", "Sep 30, 2026 · 12:00 PM"),
])
def test_chat_date_converts_to_local_time(utc_time, expected):
    from datetime import datetime, timedelta, timezone

    from lclaude.ui import format_chat_date

    class LocalDatetime(datetime):
        def astimezone(self, tz=None):
            return super().astimezone(timezone(timedelta(hours=-7)))

    with patch("lclaude.ui.datetime", LocalDatetime):
        assert format_chat_date(utc_time) == expected
