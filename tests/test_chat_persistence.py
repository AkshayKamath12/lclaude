"""Application lifecycle integration with real session files and mocked inference."""

import json
from unittest.mock import patch

import pytest

from lclaude.cli import parse_args, run_chat_loop
from lclaude.engine import InferenceEngine, OllamaEngineError
from lclaude.persistence import ChatStore, SessionStorageError
from lclaude.session import Session
from lclaude.ui import StreamAbortedError


@pytest.fixture
def store(tmp_path):
    return ChatStore(tmp_path)


@pytest.fixture
def session():
    return Session("Original guidance")


def run_loop(store, session, inputs, responses):
    engine = InferenceEngine(model="old", num_ctx=8192)
    with (
        patch("lclaude.ui.InputReader.read", side_effect=[*inputs, None]),
        patch("lclaude.ui.render_stream", side_effect=responses),
        patch("signal.signal"),
    ):
        run_chat_loop(engine, ["old", "new"], store=store, session=session)
    return engine


def test_completed_turn_and_model_selection_saved(store, session):
    run_loop(store, session, ["  complete\nquestion\n", "/model new"], ["whole answer"])
    restored, model = store.load(session.session_id)
    assert restored.conversation == session.conversation
    assert model == "new"
    assert restored.conversation[0]["content"] == "  complete\nquestion\n"


def test_model_selection_updates_saved_model_without_full_session_save(store, session):
    session.add_message("user", "saved question")
    session.add_message("assistant", "saved answer")
    store.save(session, "old")
    with patch.object(store, "save", wraps=store.save) as full_save:
        run_loop(store, session, ["/model new"], [])
    full_save.assert_not_called()
    restored, model = store.load(session.session_id)
    assert model == "new"
    assert restored.conversation == session.conversation


@pytest.mark.parametrize("failure", [StreamAbortedError(), OllamaEngineError("broken")])
def test_failed_generation_never_saves_partial_turn(store, session, failure):
    run_loop(store, session, ["first", "failed"], ["answer", failure])
    restored, _ = store.load(session.session_id)
    assert restored.conversation == session.conversation
    assert restored.conversation == [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"},
    ]


def test_interrupted_stream_partial_text_is_not_saved(store, session):
    def interrupted():
        yield "partial text"
        raise KeyboardInterrupt()

    engine = InferenceEngine(model="old", num_ctx=8192)
    with (
        patch("lclaude.ui.InputReader.read", side_effect=["failed", None]),
        patch.object(engine, "stream_chat", return_value=interrupted()),
        patch("signal.signal"),
    ):
        run_chat_loop(engine, ["old"], store=store, session=session)
    assert session.is_empty
    assert store.list_sessions() == ([], [])


def test_save_failure_keeps_memory_and_later_success_retries_whole_history(store, session, capsys):
    save = store.save
    attempts = 0

    def flaky_save(chat, model):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SessionStorageError("disk full")
        assert chat.turn_count == 2
        save(chat, model)

    with patch.object(store, "save", side_effect=flaky_save):
        run_loop(store, session, ["first", "second"], ["one", "two"])
    assert store.load(session.session_id)[0].turn_count == 2
    assert "Session not saved" not in capsys.readouterr().err


def test_clear_starts_unsaved_session_and_retains_original(store, session, capsys):
    original_id = session.session_id
    run_loop(store, session, ["first", "/clear"], ["answer"])
    assert session.session_id != original_id
    assert store.load(original_id)[0].turn_count == 1
    assert not (store.directory / f"{session.session_id}.json").exists()
    assert session.is_empty
    assert session.system_prompt == "Original guidance"
    assert len(store.list_sessions()[0]) == 1


def test_clear_proceeds_if_saving_original_fails(store, session, capsys):
    original_id = session.session_id
    with patch.object(store, "save", side_effect=SessionStorageError("no space")):
        run_loop(store, session, ["first", "/clear"], ["answer"])
    assert session.turn_count == 0
    assert session.session_id != original_id
    assert capsys.readouterr().err == ""


def test_clear_does_not_try_to_save_empty_chat(
    store, session, capsys
):
    original_id = session.session_id
    save = store.save

    def fail_empty(chat, model):
        if chat.is_empty:
            raise AssertionError("Empty chats must not be saved")
        save(chat, model)

    with patch.object(store, "save", side_effect=fail_empty):
        run_loop(store, session, ["first", "/clear", "next"], ["one", "two"])
    assert store.load(original_id)[0].conversation[0]["content"] == "first"
    assert store.load(session.session_id)[0].conversation[0]["content"] == "next"
    assert session.turn_count == 1
    assert "Session not saved" not in capsys.readouterr().err



@pytest.mark.parametrize("current_instructions", [b"Changed instructions", b"\xff", None])
def test_picker_restores_model_but_keeps_launch_instructions(
    store, session, current_instructions, capsys
):
    session.add_message("user", "earlier")
    session.add_message("assistant", "answer")
    store.save(session, "new")
    if current_instructions is not None:
        (store.project / "AGENTS.md").write_bytes(current_instructions)
    active = Session("Current instructions")
    with patch("lclaude.ui.choose_chat", return_value=session.session_id):
        engine = run_loop(store, active, ["/chat", "next"], ["done"])
    restored, model = store.load(session.session_id)
    assert engine.model == model == "new"
    assert restored.turn_count == 2
    assert restored.system_prompt is None
    output = capsys.readouterr().out
    assert "earlier" in output and "answer" in output
    assert "current instruction files are not reloaded" not in output
    assert "with saved" not in output


def test_chat_lists_newest_first_and_silently_skips_invalid_files(
    store, session, tmp_path, capsys
):
    session.add_message("user", "older")
    session.add_message("assistant", "answer")
    store.save(session, "old")
    newer = Session()
    newer.add_message("user", "newer")
    newer.add_message("assistant", "answer")
    store.save(newer, "new")
    other = ChatStore(tmp_path / "other")
    other.save(newer, "new")
    (store.directory / "bad.json").write_text("broken")
    with patch("lclaude.ui.choose_chat", return_value=None) as picker:
        run_loop(store, Session(), ["/chat"], [])
    choices = picker.call_args.args[0]
    assert [choice[0] for choice in choices] == [newer.session_id, session.session_id]
    assert "newer" in choices[0][1]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure", ["missing_model", "unsupported", "missing_file"])
def test_failed_selection_preserves_active_chat_and_model(store, session, failure, capsys):
    target = Session("Other instructions")
    target.add_message("user", "other")
    target.add_message("assistant", "answer")
    store.save(target, "missing" if failure == "missing_model" else "new")
    path = store.directory / f"{target.session_id}.json"

    def select(*args):
        if failure == "unsupported":
            data = json.loads(path.read_text())
            data["schema_version"] = 99
            path.write_text(json.dumps(data))
        elif failure == "missing_file":
            path.unlink()
        return target.session_id

    with patch("lclaude.ui.choose_chat", side_effect=select):
        engine = run_loop(store, session, ["first", "/chat", "next"], ["one", "two"])
    assert engine.model == "old"
    assert session.turn_count == 2
    assert session.system_prompt == "Original guidance"
    assert "Chat selection failed" in capsys.readouterr().err
    if failure == "unsupported":
        assert json.loads(path.read_text())["schema_version"] == 99


def test_failed_save_does_not_report_or_prevent_opening_chat_picker(store, session, capsys):
    with (
        patch.object(store, "save", side_effect=SessionStorageError("full")),
        patch("lclaude.ui.choose_chat", return_value=None) as picker,
    ):
        run_loop(store, session, ["first", "/chat"], ["answer"])
    picker.assert_called_once()
    assert session.turn_count == 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("selection", [None, "current"])
def test_cancel_or_current_selection_preserves_active_chat(store, session, selection):
    selected_id = session.session_id if selection else None
    with patch("lclaude.ui.choose_chat", return_value=selected_id):
        engine = run_loop(store, session, ["first", "/chat", "next"], ["one", "two"])
    assert engine.model == "old"
    assert session.turn_count == 2


def test_empty_chats_never_saved_by_commands(store, session):
    with patch.object(store, "save", side_effect=AssertionError("empty save")):
        run_loop(store, session, ["/model new", "/clear", "/clear", "/chat"], [])
    assert store.list_sessions() == ([], [])


@pytest.mark.parametrize("args", [["--resume", "id"], ["--list-sessions"]])
def test_chat_startup_flags_removed(args):
    with patch("sys.argv", ["lclaude", *args]), pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("has_active_chat", [False, True])
def test_open_chat_replays_history_without_generation_or_changing_saved_file(
    store, session, has_active_chat, capsys
):
    session.add_message("user", "saved question\n  details")
    session.add_message("assistant", "saved response")
    store.save(session, "new")
    path = store.directory / f"{session.session_id}.json"
    before = path.read_bytes()
    active = Session("current instructions")
    if has_active_chat:
        active.add_message("user", "different question")
        active.add_message("assistant", "different answer")
    engine = InferenceEngine(model="old", num_ctx=8192)
    with (
        patch("lclaude.ui.choose_chat", return_value=session.session_id),
        patch("lclaude.ui.InputReader.read", side_effect=["/chat", None]),
        patch.object(engine, "stream_chat") as generate,
        patch("signal.signal"),
    ):
        run_chat_loop(engine, ["old", "new"], session=active, store=store)
    generate.assert_not_called()
    output = capsys.readouterr().out
    assert "saved question\n  details" in output
    assert "saved response" in output
    resumed_output = output.split(f"Resumed chat {session.session_id}", 1)[1]
    assert "different question" not in resumed_output
    assert "current instruction files" not in output
    assert path.read_bytes() == before
    assert engine.model == "new"


@pytest.mark.parametrize("filename, content", [
    ("AGENTS.md", "Updated uppercase guidance"),
    ("agents.md", "Updated lowercase guidance"),
    (None, None),
    ("AGENTS.md", "  \n"),
])
def test_launch_instructions_apply_to_resumed_requests_and_clear(
    store, session, monkeypatch, filename, content
):
    from lclaude.cli import main
    from lclaude.instructions import DEFAULT_SYSTEM_PROMPT

    monkeypatch.chdir(store.project)
    session.add_message("user", "earlier")
    session.add_message("assistant", "old answer")
    store.save(session, "new")
    if filename:
        (store.project / filename).write_text(content, encoding="utf-8")
    expected = content if content and content.strip() else DEFAULT_SYSTEM_PROMPT
    with (
        patch("sys.argv", ["lclaude"]),
        patch("lclaude.engine.ollama.Client") as client,
        patch("lclaude.ui.choose_chat", return_value=session.session_id),
        patch("lclaude.ui.InputReader.read", side_effect=[
            "/chat", "next", "/clear", "fresh", "/chat", "again", None,
        ]),
        patch("signal.signal"),
    ):
        client.return_value.list.return_value = {"models": [{"model": "new"}]}
        client.return_value.ps.return_value = {"models": []}
        client.return_value.chat.side_effect = [
            iter([{"message": {"content": "response"}}]) for _ in range(3)
        ]
        main()
    requests = [call.kwargs["messages"] for call in client.return_value.chat.call_args_list]
    assert len(requests) == 3
    assert all(request[0] == {"role": "system", "content": expected} for request in requests)
    assert requests[0][1:] == session.conversation + [{"role": "user", "content": "next"}]
    assert requests[1][1:] == [{"role": "user", "content": "fresh"}]
    restored, _ = store.load(session.session_id)
    assert restored.system_prompt is None
    assert restored.turn_count == 3
