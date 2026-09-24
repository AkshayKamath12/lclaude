"""Project instructions from startup through complete and interrupted turns."""

from pathlib import Path
from unittest.mock import patch

import pytest

from lclaude.cli import main
from lclaude.instructions import (
    DEFAULT_SYSTEM_PROMPT,
    InstructionLoadError,
    load_system_prompt,
)


@pytest.mark.parametrize("name", ["AGENTS.md", "agents.md"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_load_preserves_markdown_and_unicode(tmp_path, name, encoding):
    content = "# Project\n\n  Use pytest. Café 日本語\n"
    (tmp_path / name).write_text(content, encoding=encoding)
    assert load_system_prompt(tmp_path) == content


def test_uppercase_wins_when_names_are_distinct(tmp_path):
    upper, lower = tmp_path / "AGENTS.md", tmp_path / "agents.md"
    upper.write_text("Upper", encoding="utf-8")
    if lower.exists():
        pytest.skip("Filesystem is case-insensitive")
    lower.write_text("Lower", encoding="utf-8")
    assert load_system_prompt(tmp_path) == "Upper"


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_missing_or_blank_uses_default(tmp_path, content):
    if content is not None:
        (tmp_path / "AGENTS.md").write_text(content, encoding="utf-8")
    assert load_system_prompt(tmp_path) == DEFAULT_SYSTEM_PROMPT


def test_does_not_search_parent_or_children(tmp_path):
    launch = tmp_path / "launch"
    child = launch / "child"
    child.mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("Parent", encoding="utf-8")
    (child / "AGENTS.md").write_text("Child", encoding="utf-8")
    assert load_system_prompt(launch) == DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize("failure", ["directory", "encoding", "permission"])
def test_read_errors_identify_path_without_fallback(tmp_path, failure):
    path = tmp_path / "AGENTS.md"
    if failure == "directory":
        path.mkdir()
    elif failure == "encoding":
        path.write_bytes(b"\xff")
    if failure == "permission":
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")) as read:
            with pytest.raises(InstructionLoadError, match="denied") as error:
                load_system_prompt(tmp_path)
            read.assert_called_once()
    else:
        with pytest.raises(InstructionLoadError) as error:
            load_system_prompt(tmp_path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize("content", [None, "Use pytest."])
def test_startup_loads_launch_directory(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    if content is not None:
        (tmp_path / "AGENTS.md").write_text(content, encoding="utf-8")
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.cli.InferenceEngine") as engine,
        patch("lclaude.cli.run_chat_loop") as loop,
    ):
        engine.return_value.verify_ready.return_value = ["model"]
        main()
    loop.assert_called_once_with(
        engine.return_value, ["model"], system_prompt=content or DEFAULT_SYSTEM_PROMPT
    )


def test_startup_read_error_prevents_engine_and_input(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_bytes(b"\xff")
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.cli.InferenceEngine") as engine,
        patch("lclaude.ui.InputReader") as reader,
        pytest.raises(SystemExit) as error,
    ):
        main()
    assert error.value.code == 1
    assert str(tmp_path / "AGENTS.md") in capsys.readouterr().err
    engine.assert_not_called()
    reader.assert_not_called()


def test_loaded_prompt_survives_turns_clear_model_change_and_interrupt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "AGENTS.md"
    path.write_text("Use pytest.", encoding="utf-8")

    def interrupted():
        yield {"message": {"content": "partial"}}
        raise KeyboardInterrupt()

    inputs = iter(["first", "second", "/clear", "/model new", "failed", "retry"])

    def read_redirected_input(prompt):
        # Editing the source after startup must not change this session's prompt.
        path.write_text("Changed on disk", encoding="utf-8")
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    with (
        patch("sys.argv", ["lclaude", "--model", "old"]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("builtins.input", side_effect=read_redirected_input),
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.stdout.isatty", return_value=False),
        patch("signal.signal"),
    ):
        client.return_value.list.return_value = {
            "models": [{"model": "old"}, {"model": "new"}]
        }
        chat = client.return_value.chat
        chat.side_effect = [
            iter([{"message": {"content": "one"}}]),
            iter([{"message": {"content": "two"}}]),
            interrupted(),
            iter([{"message": {"content": "done"}}]),
        ]
        main()
    system = {"role": "system", "content": "Use pytest."}
    requests = [call.kwargs["messages"] for call in chat.call_args_list]
    assert requests == [
        [system, {"role": "user", "content": "first"}],
        [system, {"role": "user", "content": "first"},
         {"role": "assistant", "content": "one"}, {"role": "user", "content": "second"}],
        [system, {"role": "user", "content": "failed"}],
        [system, {"role": "user", "content": "retry"}],
    ]
    assert [call.kwargs["model"] for call in chat.call_args_list] == [
        "old", "old", "new", "new"
    ]
    assert "Generation aborted" in capsys.readouterr().out
