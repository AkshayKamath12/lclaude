"""Unit tests for the CLI presentation layer and REPL loop."""

import io
import unittest
from unittest.mock import MagicMock, patch

import pytest

from lclaude.cli import handle_slash_command, main, run_chat_loop
from lclaude.commands import COMMANDS
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
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
        self.mock_engine.num_predict = 2048
        self.mock_engine.context_limit.return_value = 8192
        self.mock_engine.last_usage = None
        signal_patch = patch("signal.signal")
        signal_patch.start()
        self.addCleanup(signal_patch.stop)

    def test_normal_chat_turn_records_history(self) -> None:
        """Simulates a prompt submission followed by an EOF exit."""
        self.mock_engine.stream_chat.return_value = iter(["Hello", " world", "!"])

        with patch("lclaude.ui.InputReader.read", side_effect=["Hi", None]):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                run_chat_loop(self.mock_engine, [self.mock_engine.model])

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
            with patch("lclaude.ui.InputReader.read", side_effect=["Write code", None]):
                with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                    run_chat_loop(self.mock_engine, [self.mock_engine.model])

                    output = mock_stdout.getvalue()
                    self.assertIn("[Generation aborted by user]", output)
                    mock_rollback.assert_called_once()

    @patch("signal.signal")
    def test_keyboard_interrupt_at_prompt_ignores_subsequent_sigint(
        self, mock_signal: MagicMock
    ) -> None:
        """Verifies SIGINT is set to SIG_IGN on prompt interrupt to protect shutdown."""
        import signal

        with patch("lclaude.ui.InputReader.read", return_value=None):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                run_chat_loop(self.mock_engine, [self.mock_engine.model])

                mock_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)
                self.assertIn("Session terminated by user.", mock_stdout.getvalue())

    def test_connection_error_during_stream_rolls_back_history(self) -> None:
        """Removes pending user prompt if an engine error occurs mid-stream."""
        self.mock_engine.stream_chat.side_effect = OllamaConnectionError("Daemon dropped")

        with patch("lclaude.cli.Session.rollback") as mock_rollback:
            with patch("lclaude.ui.InputReader.read", side_effect=["Ping", None]):
                with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                    with patch("sys.stdout", new_callable=io.StringIO):
                        run_chat_loop(self.mock_engine, [self.mock_engine.model])

                        self.assertIn("[Connection Error]: Daemon dropped", mock_stderr.getvalue())
                        mock_rollback.assert_called_once()


class TestCLIMainStartup(unittest.TestCase):
    """Verifies startup lifecycle and engine pre-flight readiness checks."""

    def setUp(self) -> None:
        loader = patch("lclaude.cli.load_system_prompt", return_value="Project guidance")
        loader.start()
        self.addCleanup(loader.stop)

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
        mock_instance.verify_ready.return_value = ["model"]

        with patch("sys.argv", ["lclaude"]):
            main()

            mock_instance.verify_ready.assert_called_once_with(allow_fallback=True)
            mock_run_loop.assert_called_once_with(
                mock_instance, ["model"], system_prompt="Project guidance"
            )


@pytest.mark.parametrize("argv, installed, expected", [
    ([], ["zeta:latest", "alpha:latest"], "alpha:latest"),
    ([], ["alpha:latest", "qwen2.5:7b-instruct"], "qwen2.5:7b-instruct"),
    ([], ["alpha:latest", "qwen2.5:7b-instruct:latest"], "qwen2.5:7b-instruct"),
    (["--model", "zeta:latest"], ["alpha:latest", "zeta:latest"], "zeta:latest"),
])
def test_startup_selects_model_and_reuses_catalog(argv, installed, expected, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch("sys.argv", ["lclaude", *argv]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.cli.run_chat_loop") as loop,
    ):
        client.return_value.list.return_value = {
            "models": [{"model": name} for name in installed],
        }
        main()
        engine, models = loop.call_args.args
        assert engine.model == expected
        assert models == sorted(installed)
        client.return_value.list.assert_called_once()
        client.return_value.chat.return_value = iter([])
        list(engine.stream_chat([]))
        assert client.return_value.chat.call_args.kwargs["model"] == expected


@pytest.mark.parametrize("argv, installed", [
    ([], []),
    (["--model", "missing"], ["alpha:latest"]),
    (["-m", "qwen2.5:7b-instruct"], ["alpha:latest"]),
])
def test_startup_rejects_empty_catalog_or_missing_explicit_model(
    argv, installed, capsys, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    with (
        patch("sys.argv", ["lclaude", *argv]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.cli.run_chat_loop") as loop,
    ):
        client.return_value.list.return_value = {
            "models": [{"model": name} for name in installed],
        }
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
        loop.assert_not_called()
        assert "ollama pull" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["/help", "  /HELP  ", "\n /history\n", "/help\n/exit"])
def test_multiline_and_padded_commands_never_reach_inference(command):
    engine = MagicMock(spec=InferenceEngine)
    engine.model, engine.host = "model", "host"
    engine.num_predict = 2048
    engine.context_limit.return_value = 8192
    engine.last_usage = None
    session = Session()
    with (
        patch("lclaude.ui.InputReader") as reader_factory,
        patch("lclaude.cli.Session", return_value=session),
        patch("signal.signal"),
        patch("sys.stdout", new_callable=io.StringIO) as output,
    ):
        reader_factory.return_value.read.side_effect = [" \n ", command, None]
        run_chat_loop(engine, [engine.model])
        reader_factory.assert_called_once_with(
            commands={name: info["desc"] for name, info in COMMANDS.items()}, models=[engine.model]
        )
        engine.stream_chat.assert_not_called()
        assert session.messages == []
        if command == "/help\n/exit":
            assert "Unknown command" in output.getvalue()


@pytest.mark.parametrize(
    "failure", [KeyboardInterrupt(), OllamaConnectionError("lost"), OllamaEngineError("failed")]
)
def test_multiline_failed_turn_rolls_back_and_next_turn_succeeds(failure):
    engine = MagicMock(spec=InferenceEngine)
    engine.model, engine.host = "model", "host"
    engine.num_predict = 2048
    engine.context_limit.return_value = 8192
    engine.last_usage = None
    session = Session()
    session.add_message("user", "earlier")
    session.add_message("assistant", "answer")
    baseline = session.messages
    failed_prompt = "  failed\n    prompt\n"
    next_prompt = "  next\n    prompt\n"

    def fail_stream():
        yield "partial output"
        raise failure

    engine.stream_chat.side_effect = [fail_stream(), iter(["complete"])]
    with (
        patch("lclaude.ui.InputReader") as reader_factory,
        patch("lclaude.cli.Session", return_value=session),
        patch("signal.signal"),
        patch("sys.stdout", new_callable=io.StringIO),
        patch("sys.stderr", new_callable=io.StringIO),
    ):
        reader_factory.return_value.read.side_effect = [failed_prompt, next_prompt, None]
        run_chat_loop(engine, [engine.model])
        reader_factory.assert_called_once_with(
            commands={name: info["desc"] for name, info in COMMANDS.items()}, models=[engine.model]
        )
    assert engine.stream_chat.call_args_list[0].args[0] == baseline + [
        {"role": "user", "content": failed_prompt}
    ]
    assert engine.stream_chat.call_args_list[1].args[0] == baseline + [
        {"role": "user", "content": next_prompt}
    ]
    assert session.messages == baseline + [
        {"role": "user", "content": next_prompt},
        {"role": "assistant", "content": "complete"},
    ]


def test_clear_reuses_input_reader_but_clears_conversation():
    engine = MagicMock(spec=InferenceEngine)
    engine.model, engine.host = "model", "host"
    engine.num_predict = 2048
    engine.context_limit.return_value = 8192
    engine.last_usage = None
    engine.stream_chat.side_effect = [iter(["one"]), iter(["two"])]
    with (
        patch("lclaude.ui.InputReader") as reader_factory,
        patch("signal.signal"),
        patch("sys.stdout", new_callable=io.StringIO),
    ):
        reader_factory.return_value.read.side_effect = ["first", "/clear", "second", None]
        run_chat_loop(engine, [engine.model])
        reader_factory.assert_called_once_with(
            commands={name: info["desc"] for name, info in COMMANDS.items()}, models=[engine.model]
        )
    assert engine.stream_chat.call_args_list[1].args[0] == [
        {"role": "user", "content": "second"}
    ]


if __name__ == "__main__":
    unittest.main()
