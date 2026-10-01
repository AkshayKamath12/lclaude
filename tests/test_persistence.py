"""Durable full transcripts, validation, project boundaries, and atomic writes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lclaude.persistence import ChatStore, SessionStorageError
from lclaude.session import Session


@pytest.fixture
def store(tmp_path):
    return ChatStore(tmp_path / "project", tmp_path / "chats")


@pytest.fixture
def chat():
    session = Session("# Instructions\n日本語")
    session.add_message("user", "  whole\nquestion\n" + "x" * 100_000)
    session.add_message("assistant", "complete\nanswer\n")
    return session


def test_round_trip_full_content_and_schema(store, chat):
    store.save(chat, "model:7b")
    restored, model = store.load(chat.session_id)
    assert restored.conversation == chat.conversation
    assert restored.system_prompt is None
    assert restored.session_id == chat.session_id
    assert restored.created_at == chat.created_at
    assert restored.updated_at == chat.updated_at
    assert model == "model:7b"
    data = json.loads((store.directory / f"{chat.session_id}.json").read_text())
    assert set(data) == {"schema_version", "session_id", "project", "created_at", "updated_at",
                         "model", "messages"}
    assert data["project"] == {"path": str(store.project)}
    assert data["schema_version"] == 1
    assert data["messages"] == chat.conversation


def test_project_isolation_even_for_same_directory_name(store, chat, tmp_path):
    other = ChatStore(tmp_path / "other" / "project", tmp_path / "chats")
    store.save(chat, "model")
    assert other.directory != store.directory
    assert other.list_sessions() == ([], [])
    with pytest.raises(SessionStorageError):
        other.load(chat.session_id)
    other.directory.mkdir(parents=True)
    (other.directory / f"{chat.session_id}.json").write_bytes(
        (store.directory / f"{chat.session_id}.json").read_bytes()
    )
    with pytest.raises(SessionStorageError, match="different project"):
        other.load(chat.session_id)


def test_sessions_with_same_timestamp_sort_by_session_id(store):
    sessions = [Session(), Session()]
    for session in sessions:
        session.add_message("user", "question")
        session.add_message("assistant", "answer")
        store.save(session, "model")

    same_timestamp = max(session.created_at for session in sessions)
    for session in sessions:
        path = store.directory / f"{session.session_id}.json"
        data = json.loads(path.read_text())
        data["updated_at"] = same_timestamp
        path.write_text(json.dumps(data))

    restored, errors = store.list_sessions()
    assert errors == []
    assert [session.session_id for session, _ in restored] == sorted(
        (session.session_id for session in sessions), reverse=True
    )


@pytest.mark.parametrize("invalid, error", [
    ("{", "Cannot load"),
    ("null", "JSON object"),
    ('{"schema_version": 2}', "Unsupported schema_version"),
    ('{"schema_version": true}', "Unsupported schema_version"),
    ('{"schema_version": 1}', "Invalid schema"),
])
def test_invalid_files_rejected_and_never_overwritten(store, chat, invalid, error):
    store.directory.mkdir(parents=True)
    path = store.directory / f"{chat.session_id}.json"
    path.write_text(invalid)
    with pytest.raises(SessionStorageError, match=error):
        store.load(chat.session_id)
    with pytest.raises(SessionStorageError):
        store.save(chat, "model")
    assert path.read_text() == invalid
    saved, errors = store.list_sessions()
    assert saved == [] and len(errors) == 1


@pytest.mark.parametrize("field, value, error", [
    ("messages", [{"role": "user", "content": "pending"}], "complete"),
    ("messages", [{"role": "assistant", "content": "a"}] * 2, "role"),
    ("messages", [{"role": "user", "content": 1}, {"role": "assistant", "content": "a"}],
     "string content"),
    ("created_at", "2026-01-01T00:00:00", "timezone"),
    ("updated_at", "yesterday", "timestamp"),
    ("session_id", "../../outside", "UUID"),
    ("model", "", "model"),
])
def test_schema_validation(store, chat, field, value, error):
    store.save(chat, "model")
    path = store.directory / f"{chat.session_id}.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(SessionStorageError, match=error):
        store.load(chat.session_id)


def test_pending_turn_cannot_be_saved(store, chat):
    store.save(chat, "model")
    chat.add_message("user", "unfinished")
    with pytest.raises(SessionStorageError, match="complete"):
        store.save(chat, "model")
    assert store.load(chat.session_id)[0].turn_count == 1


@pytest.mark.parametrize("operation", ["json.dump", "os.fsync", "os.replace"])
def test_failed_write_keeps_prior_snapshot_and_cleans_temp(store, chat, operation):
    store.save(chat, "model")
    before = (store.directory / f"{chat.session_id}.json").read_bytes()
    updated = chat.updated_at
    chat.add_message("user", "next")
    chat.add_message("assistant", "answer")
    with patch(f"lclaude.persistence.{operation}", side_effect=OSError("disk failed")):
        with pytest.raises(SessionStorageError, match="disk failed"):
            store.save(chat, "model")
    assert (store.directory / f"{chat.session_id}.json").read_bytes() == before
    assert chat.updated_at == updated
    assert chat.turn_count == 2
    assert list(store.directory.glob("*.tmp")) == []
    store.save(chat, "model")
    assert store.load(chat.session_id)[0].turn_count == 2


def test_save_flushes_before_fsync_and_replace(store, chat):
    import os

    original_fsync, original_replace = os.fsync, os.replace
    events = []

    def fsync(fd):
        # JSON is already flushed to the file when fsync is called.
        temp = next(store.directory.glob("*.tmp"))
        assert json.loads(temp.read_text())["messages"] == chat.conversation
        events.append("fsync")
        original_fsync(fd)

    def replace(source, destination):
        assert Path(source).parent == Path(destination).parent == store.directory
        events.append("replace")
        original_replace(source, destination)

    with patch("lclaude.persistence.os.fsync", side_effect=fsync), patch(
        "lclaude.persistence.os.replace", side_effect=replace
    ):
        store.save(chat, "model")
    assert events == ["fsync", "replace"]


def test_clear_creates_new_identity_retaining_old_snapshot(store, chat):
    store.save(chat, "model")
    old_id, original = chat.session_id, chat.conversation
    chat.clear()
    store.save(chat, "model")
    assert chat.session_id != old_id
    assert store.load(old_id)[0].conversation == original
    assert not (store.directory / f"{chat.session_id}.json").exists()
    assert chat.is_empty
    assert [entry[0].session_id for entry in store.list_sessions()[0]] == [old_id]


def test_empty_save_does_not_create_directory(store):
    store.save(Session(), "model")
    assert not store.directory.exists()


def test_unchanged_save_preserves_timestamp_and_file(store, chat):
    store.save(chat, "model")
    path = store.directory / f"{chat.session_id}.json"
    before, timestamp = path.read_bytes(), chat.updated_at
    with patch("lclaude.persistence.os.replace", side_effect=AssertionError("unchanged")):
        store.save(chat, "model")
    assert path.read_bytes() == before
    assert chat.updated_at == timestamp


def test_save_model_updates_metadata_without_changing_history_or_timestamps(store, chat):
    store.save(chat, "old-model")
    path = store.directory / f"{chat.session_id}.json"
    before = json.loads(path.read_text())

    store.save_model(chat.session_id, "new-model")

    after = json.loads(path.read_text())
    assert after["model"] == "new-model"
    assert after["messages"] == before["messages"]
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] == before["updated_at"]


def test_old_empty_files_are_hidden_without_deleting_them(store, chat):
    store.save(chat, "model")
    path = store.directory / f"{chat.session_id}.json"
    data = json.loads(path.read_text())
    data["messages"] = []
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    assert store.list_sessions() == ([], [])
    assert path.read_bytes() == before


def test_draft_instruction_field_is_ignored_and_removed_on_save(store, chat):
    store.save(chat, "model")
    path = store.directory / f"{chat.session_id}.json"
    data = json.loads(path.read_text())
    data["system_prompt"] = {"source": "AGENTS.md", "content": "Outdated instructions"}
    path.write_text(json.dumps(data))
    restored, model = store.load(chat.session_id)
    assert restored.system_prompt is None
    assert restored.conversation == chat.conversation
    store.save(restored, model)
    assert "system_prompt" not in json.loads(path.read_text())


def test_instruction_changes_do_not_affect_saved_chat(store, chat):
    store.save(chat, "model")
    path = store.directory / f"{chat.session_id}.json"
    before = path.read_bytes()
    chat.system_prompt = "Updated instructions"
    store.save(chat, "model")
    assert path.read_bytes() == before
