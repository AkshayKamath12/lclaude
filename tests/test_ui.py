"""Unit tests for terminal presentation, stream rendering, and user input in ui.py."""

import io
import unittest
from unittest.mock import patch

import pytest

from lclaude.engine import OllamaEngineError
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


def test_replay_plain_text_preserves_full_messages_and_order(capsys):
    from lclaude.ui import render_conversation

    prompt = "  [bold]literal[/bold]\n    indented\n" + "x" * 1000
    reply = "## Result\n```python\nprint('hello')\n```"
    messages = [
        {"role": "system", "content": "hidden guidance"},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "last answer"},
    ]
    with patch("lclaude.ui.render_stream", side_effect=AssertionError("no inference replay")):
        render_conversation(messages, chat_id="saved-chat")
    output = capsys.readouterr().out
    assert output == (
        f"\nResumed chat saved-chat\n\nYou:\n{prompt}\n\nAssistant:\n{reply}\n\n"
        "You:\nfollow up\n\nAssistant:\nlast answer\n\n"
    )


def test_replay_terminal_uses_markdown_and_literal_user_text():
    from rich.markdown import Markdown

    from lclaude.ui import render_conversation

    with patch("sys.stdout.isatty", return_value=True), patch("lclaude.ui.Console") as factory:
        render_conversation([
            {"role": "user", "content": "[bold]literal[/bold]\n  code"},
            {"role": "assistant", "content": "```python\nprint(1)\n```"},
        ], chat_id="chat")
    calls = factory.return_value.print.call_args_list
    assert calls[1].args == ("[bold]literal[/bold]\n  code",)
    assert calls[1].kwargs == {"markup": False, "highlight": False}
    assert isinstance(calls[3].args[0], Markdown)
    assert "print(1)" in calls[3].args[0].markup


def test_interrupted_transcript_render_returns_to_active_chat(capsys):
    from lclaude.ui import render_conversation

    def messages():
        yield {"role": "user", "content": "first"}
        raise KeyboardInterrupt()

    render_conversation(messages(), chat_id="chat")
    assert "Chat is still active" in capsys.readouterr().out
