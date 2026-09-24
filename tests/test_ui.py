"""Unit tests for terminal presentation, stream rendering, and user input in ui.py."""

import io
import unittest
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.live import Live

from lclaude.engine import OllamaConnectionError, OllamaEngineError
from lclaude.ui import (
    InputReader,
    StreamAbortedError,
    print_aborted,
    print_banner,
    print_error,
    print_session_end,
    print_startup_error,
    render_stream,
)


class TestGetUserInput(unittest.TestCase):
    """Verifies the non-TTY input path and termination traps."""

    def setUp(self) -> None:
        tty = patch("sys.stdin.isatty", return_value=False)
        tty.start()
        self.addCleanup(tty.stop)

    @patch("builtins.input", return_value="hello world")
    def test_get_user_input_success(self, mock_input) -> None:
        result = InputReader().read()
        self.assertEqual(result, "hello world")
        mock_input.assert_called_once_with("\n> ")

    @patch("builtins.input", return_value="   padded prompt text   \n")
    def test_get_user_input_strips_whitespace(self, mock_input) -> None:
        result = InputReader().read()
        self.assertEqual(result, "padded prompt text")

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_get_user_input_keyboard_interrupt_returns_none(self, mock_input) -> None:
        result = InputReader().read()
        self.assertIsNone(result)

    @patch("builtins.input", side_effect=EOFError)
    def test_get_user_input_eof_returns_none(self, mock_input) -> None:
        result = InputReader().read()
        self.assertIsNone(result)


class TestRenderStream(unittest.TestCase):
    """Verifies streaming token delivery and mid-generation interrupt handling."""

    def test_render_stream_success(self) -> None:
        tokens = ["Here ", "is ", "a ", "code ", "snippet."]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = render_stream(tokens)

            self.assertEqual(result, "Here is a code snippet.")
            output = mock_stdout.getvalue()
            self.assertTrue(output.startswith("Assistant: "))
            self.assertIn("Here is a code snippet.", output)
            self.assertTrue(output.endswith("\n"))

    @patch("lclaude.ui.Live")
    @patch("lclaude.ui.Console")
    def test_terminal_stream_uses_rich_live_for_markdown(self, mock_console, mock_live) -> None:
        class TerminalBuffer(io.StringIO):
            def isatty(self) -> bool:
                return True

        stdout = TerminalBuffer()
        
        def tokens():
            yield "# Result\n"
            yield "print('ready')"

        with patch("sys.stdout", stdout):
            result = render_stream(tokens())

        self.assertEqual(result, "# Result\nprint('ready')")
        
        # Verifies rich.console.Console was initialized correctly
        mock_console.assert_called_once_with(file=stdout, force_terminal=True)
        
        # Verifies rich.live.Live was configured for smooth, visible streaming
        mock_live.assert_called_once()
        self.assertEqual(mock_live.call_args[1]["refresh_per_second"], 15)
        self.assertEqual(mock_live.call_args[1]["vertical_overflow"], "visible")
        
        # Verifies Live.update() was called to repaint the terminal
        self.assertGreater(mock_live.return_value.__enter__.return_value.update.call_count, 0)

    def test_nonterminal_stream_stays_plain_and_single_pass(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(render_stream(["plain", " text"]), "plain text")
            output = stdout.getvalue()

        self.assertEqual(output, "Assistant: plain text\n")
        self.assertNotIn("\x1b[", output)

    def test_interrupted_stream_does_not_replace_partial_output(self) -> None:
        def interrupted():
            yield "partial"
            raise KeyboardInterrupt

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with patch("lclaude.ui.Console") as console:
                with self.assertRaises(StreamAbortedError):
                    render_stream(interrupted())

        console.assert_not_called()
        self.assertIn("Assistant: partial", stdout.getvalue())

    def test_render_stream_empty(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = render_stream([])

            self.assertEqual(result, "")
            self.assertEqual(mock_stdout.getvalue(), "Assistant: \n")

    def test_render_stream_aborted_by_keyboard_interrupt(self) -> None:
        def interrupted_generator():
            yield "Starting "
            yield "generation"
            raise KeyboardInterrupt()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            with self.assertRaises(StreamAbortedError):
                render_stream(interrupted_generator())

            output = mock_stdout.getvalue()
            # Confirms partial tokens flushed before abort and trailing newline emitted
            self.assertIn("Assistant: Starting generation", output)


class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize("ending", [None, KeyboardInterrupt, OllamaEngineError,
                                    OllamaConnectionError])
@pytest.mark.parametrize("has_tokens", [False, True])
def test_thinking_spinner_lifecycle(ending, has_tokens):
    stdout = TerminalBuffer()
    console = Console(
        file=stdout, force_terminal=True, legacy_windows=False, _environ={"TERM": "xterm"}
    )
    status = console.status("Thinking…")
    markdown_displays = []

    def markdown_live(*args, **kwargs):
        # The spinner must be gone before the Markdown display starts.
        assert not status._live.is_started
        live = Live(*args, **kwargs)
        markdown_displays.append(live)
        return live

    def tokens():
        assert status._live.is_started
        status._live.refresh()  # Simulate an animation frame without a timed sleep.
        yield ""
        assert status._live.is_started
        if has_tokens:
            yield "first "
            assert not status._live.is_started
            assert markdown_displays[0].is_started
            yield "**second**"
        if ending is not None:
            raise ending()

    with (
        patch("sys.stdout", stdout),
        patch("lclaude.ui.Console", return_value=console),
        patch.object(console, "status", return_value=status) as make_status,
        patch("lclaude.ui.Live", side_effect=markdown_live),
    ):
        if ending is None:
            assert render_stream(tokens()) == ("first **second**" if has_tokens else "")
        else:
            expected = StreamAbortedError if ending is KeyboardInterrupt else ending
            with pytest.raises(expected):
                render_stream(tokens())

    make_status.assert_called_once_with("Thinking…")
    assert not status._live.is_started
    assert all(not live.is_started for live in markdown_displays)
    output = stdout.getvalue()
    assert "Thinking…" in output
    # Rich restores the cursor and clears its transient status display.
    assert "\x1b[?25h" in output
    assert "\x1b[2K" in output
    assert output.rfind("\x1b[?25h") > output.rfind("\x1b[?25l")
    if has_tokens:
        assert "first" in output
        assert "second" in output


@pytest.mark.parametrize("ending", [None, KeyboardInterrupt, OllamaEngineError])
@pytest.mark.parametrize("has_tokens", [False, True])
def test_redirected_stream_never_starts_spinner(ending, has_tokens):
    def tokens():
        if has_tokens:
            yield "plain text"
        if ending is not None:
            raise ending()

    with (
        patch("sys.stdout", new_callable=io.StringIO) as stdout,
        patch("lclaude.ui.Console") as console,
    ):
        if ending is None:
            assert render_stream(tokens()) == ("plain text" if has_tokens else "")
        else:
            expected = StreamAbortedError if ending is KeyboardInterrupt else ending
            with pytest.raises(expected):
                render_stream(tokens())

    console.assert_not_called()
    assert stdout.getvalue() == (
        "Assistant: " + ("plain text" if has_tokens else "") + ("\n" if ending is None else "")
    )


class TestTerminalDisplays(unittest.TestCase):
    """Verifies formatting of banners, notices, and error outputs."""

    def test_print_banner_includes_critical_metadata(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            print_banner(model="qwen2.5-coder:7b", host="http://localhost:11434")
            output = mock_stdout.getvalue()

            self.assertIn("lclaude - Local CLI Coding Assistant", output)
            self.assertIn("qwen2.5-coder:7b", output)
            self.assertIn("http://localhost:11434", output)
            self.assertIn("Abort generation: Ctrl+C | Exit: Ctrl+C at prompt", output)

    def test_print_aborted(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            print_aborted()
            self.assertIn("[Generation aborted by user]\n", mock_stdout.getvalue())

    def test_print_session_end(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            print_session_end()
            self.assertIn("Session terminated by user.\n", mock_stdout.getvalue())

    def test_print_error_writes_to_stderr(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            print_error(title="Connection Error", message="Failed to reach daemon")
            self.assertEqual(
                mock_stderr.getvalue(),
                "\n[Connection Error]: Failed to reach daemon\n",
            )

    def test_print_startup_error_writes_to_stderr(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            print_startup_error(message="Model 'llama3' not found locally.")
            self.assertEqual(
                mock_stderr.getvalue(),
                "Startup check failed: Model 'llama3' not found locally.\n",
            )


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("ending", [None, KeyboardInterrupt, OllamaEngineError])
def test_pinned_footer_restores_scroll_region(ending):
    from lclaude.ui import pinned_footer

    stdout = TerminalBuffer()
    console = Console(file=stdout, force_terminal=True, legacy_windows=False,
                      width=80, height=24)
    try:
        with pinned_footer(console, "Prompt ~100 / 8,192 tokens") as refresh:
            assert "\x1b[1;23r" in stdout.getvalue()
            assert "\x1b[24;1H" in stdout.getvalue()
            refresh()
            assert stdout.getvalue().count("Prompt ~100") == 1
            if ending:
                raise ending()
    except (KeyboardInterrupt, OllamaEngineError):
        pass
    assert "\x1b[r" in stdout.getvalue()
    assert stdout.getvalue().endswith("\x1b[2K\x1b8")


def test_footer_resizes_and_unknown_is_silent():
    from lclaude.ui import pinned_footer

    stdout = TerminalBuffer()
    console = Console(file=stdout, force_terminal=True, legacy_windows=False,
                      width=80, height=24)
    with pinned_footer(console, "") as refresh:
        refresh()
    assert stdout.getvalue() == ""
    with pinned_footer(console, "Prompt ~100") as refresh:
        console.height = 30
        refresh()
        assert "\x1b[1;29r" in stdout.getvalue()
        assert "\x1b[30;1H" in stdout.getvalue()


def test_input_toolbar_updates_in_place_and_hides_unknown(capsys):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lclaude.context import ContextBudget, PromptCount

    with create_pipe_input() as pipe:
        reader = InputReader(input_stream=pipe, output_stream=DummyOutput())
        reader.set_context(PromptCount(1, 2, 3), None, ContextBudget())
        assert reader._prompt.bottom_toolbar is None
        reader.set_context(PromptCount(1, 2, 3), 8192, ContextBudget())
        assert reader._prompt.bottom_toolbar == (
            "Prompt ~6 / 8,192 tokens | 2,048 reserved for reply"
        )
        reader.set_context(PromptCount(1, 200, 3), 4096, ContextBudget())
        assert "~204 / 4,096" in reader._prompt.bottom_toolbar
        assert capsys.readouterr().out == ""


def test_toolbar_row_visibility_tracks_runtime_allocation():
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lclaude.context import ContextBudget, PromptCount

    with create_pipe_input() as pipe:
        reader = InputReader(input_stream=pipe, output_stream=DummyOutput())
        prompt = reader._prompt
        # Exercise the library's actual row visibility, not just the text value.
        toolbar = prompt.layout.container.children[-1]
        with set_app(prompt.app):
            prompt.app.renderer._min_available_height = 24
            assert not toolbar.filter()
            reader.set_context(PromptCount(1, 2, 3), 4096, ContextBudget())
            assert toolbar.filter()
            # Switching to a cold model must remove the row again.
            reader.set_context(PromptCount(1, 2, 3), None, ContextBudget())
            assert not toolbar.filter()
            reader.set_context(PromptCount(1, 2, 3), 8192, ContextBudget())
            assert toolbar.filter()
