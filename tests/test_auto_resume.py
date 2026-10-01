"""Launch-directory automatic resume and startup failure behavior."""

from unittest.mock import patch

import pytest

from lclaude.cli import main
from lclaude.persistence import ChatStore
from lclaude.session import Session


def saved_chat(store, text, model="saved"):
    chat = Session("Outdated instructions")
    chat.add_message("user", text)
    chat.add_message("assistant", "answer")
    store.save(chat, model)
    return chat


@pytest.mark.parametrize("override", [False, True])
def test_launch_resumes_newest_project_chat_with_current_instructions(
    tmp_path, monkeypatch, capsys, override
):
    monkeypatch.chdir(tmp_path)
    store = ChatStore(tmp_path)
    saved_chat(store, "older")
    latest = saved_chat(store, "latest")
    saved_chat(ChatStore(tmp_path / "other"), "other project")
    (tmp_path / "AGENTS.md").write_text("Current instructions")
    before = (store.directory / f"{latest.session_id}.json").read_bytes()
    with (
        patch("sys.argv", ["lclaude", *(["--model", "override"] if override else [])]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.ui.InputReader.read", side_effect=["next", None]),
        patch("signal.signal"),
    ):
        client.return_value.list.return_value = {
            "models": [{"model": "saved"}, {"model": "override"}]
        }
        client.return_value.ps.return_value = {"models": []}
        client.return_value.chat.return_value = iter([{"message": {"content": "next answer"}}])
        main()
    request = client.return_value.chat.call_args.kwargs
    assert request["model"] == ("override" if override else "saved")
    assert request["messages"] == [
        {"role": "system", "content": "Current instructions"},
        *latest.conversation, {"role": "user", "content": "next"},
    ]
    assert store.load(latest.session_id)[0].turn_count == 2
    assert (store.directory / f"{latest.session_id}.json").read_bytes() != before
    output = capsys.readouterr().out
    assert "latest" in output and "answer" in output
    assert "other project" not in output


@pytest.mark.parametrize("valid_chat", [False, True])
def test_invalid_files_are_skipped_and_preserved(tmp_path, monkeypatch, capsys, valid_chat):
    monkeypatch.chdir(tmp_path)
    store = ChatStore(tmp_path)
    chat = saved_chat(store, "valid") if valid_chat else None
    store.directory.mkdir(parents=True, exist_ok=True)
    bad = store.directory / "broken.json"
    bad.write_text("{")
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.cli.InferenceEngine") as engine,
        patch("lclaude.cli.run_chat_loop") as loop,
    ):
        engine.return_value.verify_ready.return_value = ["saved"]
        main()
    active = loop.call_args.kwargs["session"]
    assert active.session_id == chat.session_id if chat else active.is_empty
    assert "Invalid saved session" not in capsys.readouterr().err
    assert bad.read_text() == "{"


def test_missing_saved_model_does_not_silently_fall_back(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = ChatStore(tmp_path)
    chat = saved_chat(store, "question", model="missing")
    path = store.directory / f"{chat.session_id}.json"
    before = path.read_bytes()
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.cli.run_chat_loop") as loop,
        pytest.raises(SystemExit),
    ):
        client.return_value.list.return_value = {"models": [{"model": "other"}]}
        main()
    loop.assert_not_called()
    assert "missing" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_latest_utility_ignores_empty_chats(tmp_path):
    store = ChatStore(tmp_path)
    assert store.latest_session() == (None, [])
    chat = saved_chat(store, "question")
    store.save(Session(), "saved")
    latest, errors = store.latest_session()
    assert latest[0].session_id == chat.session_id
    assert not errors


def test_resume_without_new_turn_does_not_write_or_generate(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = ChatStore(tmp_path)
    chat = saved_chat(store, "saved question")
    path = store.directory / f"{chat.session_id}.json"
    before = path.read_bytes()
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.ui.InputReader.read", return_value=None),
        patch("signal.signal"),
    ):
        client.return_value.list.return_value = {"models": [{"model": "saved"}]}
        client.return_value.ps.return_value = {"models": []}
        main()
    client.return_value.chat.assert_not_called()
    assert path.read_bytes() == before
    assert "saved question" in capsys.readouterr().out
