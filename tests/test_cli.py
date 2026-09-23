"""Unit tests for the CLI presentation layer and REPL loop."""

import io
import unittest
from unittest.mock import MagicMock, patch

from lclaude.cli import handle_slash_command, main, run_chat_loop
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
)
from lclaude.session import Session


class TestCLISlashCommands(unittest.TestCase):
    """Verifies local command handling before inference calls."""

    def test_clear_command_resets_history(self) -> None:
        session = Session()
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/clear", session)

            self.assertTrue(handled)
            self.assertTrue(session.is_empty)
            self.assertIn("Cleared", mock_stdout.getvalue())

    def test_history_command_prints_turns(self) -> None:
        session = Session()
        session.add_message("user", "Tell me a secret")

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/history", session)

            self.assertTrue(handled)
            self.assertIn("Active Context: 1 messages", mock_stdout.getvalue())
            self.assertIn("[user]: Tell me a secret", mock_stdout.getvalue())

    def test_help_command_outputs_options(self) -> None:
        session = Session()

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            handled = handle_slash_command("/help", session)

            self.assertTrue(handled)
            output = mock_stdout.getvalue()
            self.assertIn("/clear", output)
            self.assertIn("/history", output)
            self.assertIn("/exit", output)

    def test_exit_command_terminates_process(self) -> None:
        session = Session()

        with patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                handle_slash_command("/exit", session)
            self.assertEqual(ctx.exception.code, 0)


class TestCLIChatLoop(unittest.TestCase):
    """Verifies streaming output, history accumulation, and signal rollback."""

    def setUp(self) -> None:
        self.mock_engine = MagicMock(spec=InferenceEngine)
        self.mock_engine.model = "qwen2.5:7b-instruct"
        self.mock_engine.host = "http://localhost:11434"

    def test_normal_chat_turn_records_history(self) -> None:
        """Simulates a prompt submission followed by an EOF exit."""
        self.mock_engine.stream_chat.return_value = iter(["Hello", " world", "!"])

        with patch("builtins.input", side_effect=["Hi", EOFError]):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                run_chat_loop(self.mock_engine)

                output = mock_stdout.getvalue()
                self.assertIn("Assistant: Hello world!", output)
                self.assertIn("Session terminated by user.", output)

        # Engine was called with the snapshot of messages available when stream began
        self.mock_engine.stream_chat.assert_called_once()
        history_arg = self.mock_engine.stream_chat.call_args[0][0]

        self.assertEqual(len(history_arg), 1)
        self.assertEqual(history_arg[0], {"role": "user", "content": "Hi"})

    def test_ctrl_c_during_stream_triggers_state_rollback(self) -> None:
        """Traps KeyboardInterrupt during token generation and rolls back user turn."""
        def interrupted_stream(messages):
            yield "Starting output..."
            raise KeyboardInterrupt()

        self.mock_engine.stream_chat.side_effect = interrupted_stream

        with patch("lclaude.cli.Session.rollback") as mock_rollback:
            with patch("builtins.input", side_effect=["Write code", EOFError]):
                with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                    run_chat_loop(self.mock_engine)

                    output = mock_stdout.getvalue()
                    self.assertIn("[Generation aborted by user]", output)
                    mock_rollback.assert_called_once()

    @patch("signal.signal")
    def test_keyboard_interrupt_at_prompt_ignores_subsequent_sigint(
        self, mock_signal: MagicMock
    ) -> None:
        """Verifies SIGINT is set to SIG_IGN on prompt interrupt to protect shutdown."""
        import signal

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                run_chat_loop(self.mock_engine)

                mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
                self.assertIn("Session terminated by user.", mock_stdout.getvalue())

    def test_connection_error_during_stream_rolls_back_history(self) -> None:
        """Removes pending user prompt if an engine error occurs mid-stream."""
        self.mock_engine.stream_chat.side_effect = OllamaConnectionError("Daemon dropped")

        with patch("lclaude.cli.Session.rollback") as mock_rollback:
            with patch("builtins.input", side_effect=["Ping", EOFError]):
                with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                    with patch("sys.stdout", new_callable=io.StringIO):
                        run_chat_loop(self.mock_engine)

                        self.assertIn("[Connection Error]: Daemon dropped", mock_stderr.getvalue())
                        mock_rollback.assert_called_once()


class TestCLIMainStartup(unittest.TestCase):
    """Verifies startup lifecycle and engine pre-flight readiness checks."""

    @patch("lclaude.cli.InferenceEngine")
    def test_main_exits_on_readiness_failure(self, mock_engine_cls: MagicMock) -> None:
        """Exits with code 1 if verify_ready fails."""
        mock_instance = mock_engine_cls.return_value
        mock_instance.verify_ready.side_effect = ModelNotFoundError("Model missing")

        with patch("sys.argv", ["lclaude"]):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                with self.assertRaises(SystemExit) as ctx:
                    main()

                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("Startup check failed: Model missing", mock_stderr.getvalue())

    @patch("lclaude.cli.run_chat_loop")
    @patch("lclaude.cli.InferenceEngine")
    def test_main_succeeds_when_ready(
        self, mock_engine_cls: MagicMock, mock_run_loop: MagicMock
    ) -> None:
        """Launches REPL when verify_ready passes."""
        mock_instance = mock_engine_cls.return_value
        mock_instance.verify_ready.return_value = None

        with patch("sys.argv", ["lclaude"]):
            main()

            mock_instance.verify_ready.assert_called_once()
            mock_run_loop.assert_called_once_with(mock_instance)


if __name__ == "__main__":
    unittest.main()