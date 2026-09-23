"""Exercise actual editor key processing with pipe input, never an interactive terminal."""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import Mock, patch

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lclaude.commands import COMMANDS
from lclaude.ui import InputReader

UP = "\x1b[A"
DOWN = "\x1b[B"
LEFT = "\x1b[D"
NEWLINE = "\x1b\r"


def paste(text):
    return "\x1b[200~" + text + "\x1b[201~"


class Editor:
    """Bounded async harness with a key-processing barrier instead of sleeps."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.output = DummyOutput()
        self.reader = InputReader(
            commands={name: info["desc"] for name, info in COMMANDS.items()},
            input_stream=pipe, output_stream=self.output,
        )
        self.prompt = self.reader._prompt
        self.processed = asyncio.Event()
        self.prompt.key_bindings.add("f12")(lambda event: self.processed.set())

    @property
    def buffer(self):
        return self.prompt.default_buffer

    async def send(self, text):
        self.processed.clear()
        self.pipe.send_text(text + "\x1b[24~")
        await asyncio.wait_for(self.processed.wait(), 3)
        # Completion runs in a background task after key processing.
        tasks = [task for task in self.prompt.app._background_tasks
                 if "async_completer" in task.get_coro().__qualname__]
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 3)

    async def finish(self, text="\r"):
        self.pipe.send_text(text)
        result = await asyncio.wait_for(self.task, 3)
        if isinstance(result, str):
            return self.reader._restore_pasted_text(result)
        return result

    @asynccontextmanager
    async def running(self):
        ready = asyncio.Event()

        async def run():
            try:
                return await self.prompt.prompt_async(pre_run=ready.set)
            except (KeyboardInterrupt, EOFError) as exc:
                # Keep KeyboardInterrupt from escaping an asyncio task's runner.
                return exc

        self.task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(ready.wait(), 3)
            # Rendering starts the library's background history load. Await it
            # explicitly so history tests cannot race the first input event.
            await asyncio.wait_for(self.buffer._load_history_task, 3)
            yield self
        finally:
            if not self.task.done():
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass


@pytest.mark.parametrize("submit", ["\r", "\n"])
def test_single_line_submission_preserves_spaces(submit):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                assert await editor.finish("  hello  " + submit) == "  hello  "
            assert editor.prompt.history.get_strings() == ["  hello  "]
    asyncio.run(scenario())


@pytest.mark.parametrize("enter", ["\r", "\n"])
def test_multiline_newline_binding_and_separate_escape_fallback(enter):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("one\x1b" + enter + "  two")
                assert editor.buffer.text == "one\n  two"
                assert not editor.task.done()
                # Deliver Escape and Enter separately, without sleeping or
                # allowing another key to break the sequence.
                pipe.send_text("\x1b")
                await editor.send(enter + "three")
                assert editor.buffer.text == "one\n  two\nthree"
                assert await editor.finish() == "one\n  two\nthree"
    asyncio.run(scenario())


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_bracketed_paste_waits_for_submission(ending):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("prefix suffix" + LEFT * 6)
                await editor.send(paste(ending + "  code" + ending + "/exit" + ending))
                expected = "prefix \n  code\n/exit\nsuffix"
                assert editor.buffer.text == "prefix <pasted 4 lines>suffix"
                assert not editor.task.done()
                assert editor.prompt.history.get_strings() == []
                assert await editor.finish() == expected
            assert editor.prompt.history.get_strings() == [expected]
    asyncio.run(scenario())


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", " \n \t"])
def test_blank_enter_does_not_submit_or_record_history(blank):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send(paste(blank) + "\r")
                assert not editor.task.done()
                assert editor.prompt.history.get_strings() == []
                assert await editor.finish("ok\r") == blank + "ok"
    asyncio.run(scenario())


def test_history_multiline_recall_edit_and_draft_restoration():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            for text in ["first\nsecond", "/clear", "/clear"]:
                async with editor.running():
                    assert await editor.finish(paste(text) + "\r") == text
            assert editor.prompt.history.get_strings() == ["first\nsecond", "/clear"]
            async with editor.running():
                await editor.send("draft" + UP)
                assert editor.buffer.text == "/clear"
                await editor.send(UP)
                assert editor.buffer.text == "first\nsecond"
                await editor.send(UP)
                assert editor.buffer.document.cursor_position_row == 0
                await editor.send(UP)
                assert editor.buffer.text == "first\nsecond"  # oldest edge
                await editor.send(DOWN + DOWN + DOWN)
                assert editor.buffer.text == "draft"
                await editor.send(DOWN)
                assert editor.buffer.text == "draft"  # newest edge
                await editor.send(UP + " edited")
                assert await editor.finish() == "/clear edited"
            assert editor.prompt.history.get_strings() == [
                "first\nsecond", "/clear", "/clear edited"
            ]
    asyncio.run(scenario())


def test_vertical_movement_preserves_column_through_short_lines():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("abcdef" + NEWLINE + "x" + NEWLINE + "abcdef" + UP)
                assert editor.buffer.document.cursor_position_col == 1
                await editor.send(UP)
                assert editor.buffer.document.cursor_position_col == 6
                await editor.send(DOWN + DOWN)
                assert editor.buffer.document.cursor_position_col == 6
                await editor.send(LEFT * 3 + NEWLINE)
                assert await editor.finish() == "abcdef\nx\nabc\ndef"
    asyncio.run(scenario())


def test_multiline_paste_uses_atomic_backspace_and_removes_hidden_payload():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send(paste("first\nsecond"))
                assert editor.buffer.text == "<pasted 2 lines>"
                await editor.send("\x08")
                assert editor.buffer.text == ""
                assert editor.reader._pasted_blocks == []
                assert editor.prompt.history.get_strings() == []
                await editor.send("replacement")
                assert await editor.finish() == "replacement"
            assert editor.prompt.history.get_strings() == ["replacement"]
    asyncio.run(scenario())


def test_soft_wrapping_does_not_add_logical_history_boundaries():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                assert await editor.finish("previous\r") == "previous"
            async with editor.running():
                await editor.send("x" * 200 + UP)
                assert editor.buffer.text == "previous"
                await editor.send(DOWN)
                assert await editor.finish() == "x" * 200
    asyncio.run(scenario())


@pytest.mark.parametrize("draft", ["", "unfinished\ntext"])
def test_ctrl_c_discards_draft_without_history(draft):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send(paste(draft))
                assert isinstance(await editor.finish("\x03"), KeyboardInterrupt)
            assert editor.prompt.history.get_strings() == []
    asyncio.run(scenario())


def test_ctrl_d_deletes_in_nonempty_buffer_and_exits_empty_buffer():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("abc" + LEFT + "\x04")
                assert editor.buffer.text == "ab"
                await editor.send("\x04")  # at end of nonempty text: no submission
                assert not editor.task.done()
                assert await editor.finish() == "ab"
            async with editor.running():
                assert isinstance(await editor.finish("\x04"), EOFError)
    asyncio.run(scenario())


def test_windows_ctrl_z_alias():
    async def check():
        with create_pipe_input() as pipe:
            with patch("lclaude.ui.sys.platform", "win32"):
                editor = Editor(pipe)
            async with editor.running():
                await editor.send("draft\x1a")
                assert editor.buffer.text == "draft"
                assert not editor.task.done()
                assert await editor.finish() == "draft"
            async with editor.running():
                assert isinstance(await editor.finish("\x1a"), EOFError)
    asyncio.run(check())


@pytest.mark.parametrize("termination", ["\x03", "\x04", "closed", "accepted", "error"])
def test_terminal_lifecycle_restored_after_exit(termination):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            raw_mode = pipe.raw_mode
            active = []

            @contextmanager
            def tracked_raw_mode():
                with raw_mode():
                    active.append(True)
                    try:
                        yield
                    finally:
                        active.pop()

            with (
                patch.object(pipe, "raw_mode", tracked_raw_mode),
                patch.object(editor.output, "disable_bracketed_paste") as disable_paste,
                patch.object(editor.output, "show_cursor") as show_cursor,
                patch.object(editor.output, "reset_attributes") as reset,
            ):
                async with editor.running():
                    assert active
                    if termination == "closed":
                        await editor.send("unsubmitted")
                        pipe.close()
                        assert isinstance(await asyncio.wait_for(editor.task, 3), EOFError)
                    elif termination == "error":
                        editor.prompt.app.exit(exception=RuntimeError("failed"))
                        with pytest.raises(RuntimeError, match="failed"):
                            await asyncio.wait_for(editor.task, 3)
                    elif termination == "accepted":
                        assert await editor.finish("ok\r") == "ok"
                    else:
                        await editor.finish(termination)
                assert not active
                disable_paste.assert_called()
                show_cursor.assert_called()
                reset.assert_called()
            if termination != "closed":
                async with editor.running():
                    assert await editor.finish("usable\r") == "usable"
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [KeyboardInterrupt, EOFError])
def test_reader_converts_editor_interrupt_and_eof_to_none(failure):
    with create_pipe_input() as pipe:
        reader = InputReader(input_stream=pipe, output_stream=DummyOutput())
        with patch.object(reader._prompt, "prompt", side_effect=failure):
            assert reader.read() is None


def test_reader_returns_editor_text_verbatim():
    with create_pipe_input() as pipe:
        reader = InputReader(input_stream=pipe, output_stream=DummyOutput())
        with patch.object(reader._prompt, "prompt", return_value=" \n code\n"):
            assert reader.read() == " \n code\n"


def test_synchronous_reader_reuses_real_editor_history():
    with create_pipe_input() as pipe:
        reader = InputReader(input_stream=pipe, output_stream=DummyOutput())
        prompt = reader._prompt
        original_prompt = prompt.prompt

        def read_keys(keys):
            def setup():
                asyncio.get_running_loop().call_later(
                    3, lambda: prompt.app.exit(exception=TimeoutError("editor did not finish"))
                )

                async def send_when_ready():
                    await prompt.default_buffer._load_history_task
                    pipe.send_text(keys)

                prompt.app.create_background_task(send_when_ready())

            with patch.object(prompt, "prompt", side_effect=lambda: original_prompt(pre_run=setup)):
                return reader.read()

        assert read_keys("first" + NEWLINE + "second\r") == "first\nsecond"
        assert read_keys(UP + "\r") == "first\nsecond"
        assert read_keys("\x03") is None


def test_interactive_streams_select_editor():
    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("lclaude.ui.PromptSession") as prompt,
        patch("builtins.input") as line_input,
    ):
        prompt.__getitem__.return_value = prompt
        prompt.return_value.prompt.return_value = "hello\nworld"
        assert InputReader().read() == "hello\nworld"
        prompt.assert_called_once()
        line_input.assert_not_called()


@pytest.mark.parametrize("stdin_tty, stdout_tty", [(False, True), (True, False), (False, False)])
def test_non_tty_uses_line_input_without_constructing_editor(stdin_tty, stdout_tty):
    with (
        patch("sys.stdin.isatty", return_value=stdin_tty),
        patch("sys.stdout.isatty", return_value=stdout_tty),
        patch("lclaude.ui.PromptSession") as prompt,
        patch("builtins.input", side_effect=[" first ", "second", EOFError]),
    ):
        reader = InputReader()
        assert [reader.read(), reader.read(), reader.read()] == ["first", "second", None]
        prompt.assert_not_called()


def test_completion_opens_filters_and_displays_registry_descriptions():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("/")
                completions = editor.buffer.complete_state.completions
                assert [(c.text, c.display_meta_text) for c in completions] == [
                    (name, info["desc"]) for name, info in COMMANDS.items()
                ]
                await editor.send("H")
                assert [c.text for c in editor.buffer.complete_state.completions] == [
                    name for name in COMMANDS if name.startswith("/h")
                ]
                await editor.send("z")
                assert editor.buffer.complete_state is None
                await editor.send("\x08")
                assert editor.buffer.complete_state is not None
                assert await editor.finish() == "/H"
    asyncio.run(scenario())


def test_new_registry_command_completes_without_executing(monkeypatch):
    handler = Mock()
    monkeypatch.setitem(COMMANDS, "/custom", {"desc": "Custom description", "handler": handler})

    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("/cu")
                completion = editor.buffer.complete_state.completions[0]
                assert completion.display_meta_text == "Custom description"
                await editor.send("\t")
                assert editor.buffer.text == "/custom"
                assert await editor.finish() == "/custom"
                handler.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("selection", ["\t", DOWN, DOWN + DOWN + UP])
def test_completion_selection_and_enter_submission(selection):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("/h")
                expected = next(name for name in COMMANDS if name.startswith("/h"))
                await editor.send(selection)
                assert editor.buffer.text == expected
                assert not editor.task.done()
                assert await editor.finish() == expected
    asyncio.run(scenario())


def test_completion_escape_dismisses_and_history_still_works():
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                assert await editor.finish("previous\r") == "previous"
            async with editor.running():
                await editor.send("/h")
                await editor.send(DOWN + "\x1b")
                assert editor.buffer.complete_state is None
                assert editor.buffer.text == "/h"
                await editor.send(UP)
                assert editor.buffer.text == "previous"
                await editor.send(DOWN)
                assert await editor.finish() == "/h"
    asyncio.run(scenario())


@pytest.mark.parametrize("text", [
    "ordinary prose", "hello /h", " /h", "/h argument", "/h" + NEWLINE + "x",
    paste("/h"), paste("/h\ncode"), "/" + paste("h"),
])
def test_completion_stays_closed_for_prose_multiline_and_paste(text):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send(text)
                assert editor.buffer.complete_state is None
                await editor.send("\t")
                assert editor.buffer.complete_state is None
                assert not editor.task.done()
    asyncio.run(scenario())


@pytest.mark.parametrize("ending", [NEWLINE, "\x03"])
def test_open_completion_preserves_newline_and_interrupt(ending):
    async def scenario():
        with create_pipe_input() as pipe:
            editor = Editor(pipe)
            async with editor.running():
                await editor.send("/h")
                assert editor.buffer.complete_state is not None
                if ending == NEWLINE:
                    await editor.send(ending)
                    assert editor.buffer.complete_state is None
                    assert await editor.finish() == "/h\n"
                else:
                    assert isinstance(await editor.finish(ending), KeyboardInterrupt)
                    assert editor.prompt.history.get_strings() == []
    asyncio.run(scenario())
