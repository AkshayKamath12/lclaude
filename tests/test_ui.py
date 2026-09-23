"""Unit tests for terminal presentation, stream rendering, and user input in ui.py."""

import io
import unittest
from unittest.mock import patch

from lclaude.ui import (
    StreamAbortedError,
    get_user_input,
    print_aborted,
    print_banner,
    print_error,
    print_session_end,
    print_startup_error,
    render_stream,
)


class TestGetUserInput(unittest.TestCase):
    """Verifies user input prompt handling and termination traps."""

    @patch("builtins.input", return_value="hello world")
    def test_get_user_input_success(self, mock_input) -> None:
        result = get_user_input()
        self.assertEqual(result, "hello world")
        mock_input.assert_called_once_with("\n> ")

    @patch("builtins.input", return_value="   padded prompt text   \n")
    def test_get_user_input_strips_whitespace(self, mock_input) -> None:
        result = get_user_input()
        self.assertEqual(result, "padded prompt text")

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_get_user_input_keyboard_interrupt_returns_none(self, mock_input) -> None:
        result = get_user_input()
        self.assertIsNone(result)

    @patch("builtins.input", side_effect=EOFError)
    def test_get_user_input_eof_returns_none(self, mock_input) -> None:
        result = get_user_input()
        self.assertIsNone(result)


class TestRenderStream(unittest.TestCase):
    """Verifies streaming token delivery and mid-generation interrupt handling."""

    def test_render_stream_success(self) -> None:
        tokens = ["Here ", "is ", "a ", "code ", "snippet."]

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = render_stream(tokens)

            self.assertEqual(result, "Here is a code snippet.")
            output = mock_stdout.getvalue()
            self.assertTrue(output.startswith("\nAssistant: "))
            self.assertIn("Here is a code snippet.", output)
            self.assertTrue(output.endswith("\n"))

    def test_render_stream_empty(self) -> None:
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = render_stream([])

            self.assertEqual(result, "")
            self.assertEqual(mock_stdout.getvalue(), "\nAssistant: \n")

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
            self.assertIn("\nAssistant: Starting generation\n", output)


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