"""Unit tests for terminal presentation, stream rendering, and user input in ui.py."""

import io
import unittest
from unittest.mock import patch

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