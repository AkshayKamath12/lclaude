"""Versioned command transcripts, including older read-only and plain chats."""

import json

import pytest

from lclaude.persistence import ChatStore, SessionStorageError, deserialize, serialize
from lclaude.session import Session


def tool_call(name="run_command", call_id=None):
    result = {"function": {"name": name, "arguments": {"command": "echo hi", "shell": "sh"}}}
    if call_id:
        result["id"] = call_id
    return result


def session_with_tools():
    session = Session()
    session.add_message("user", "inspect")
    start = session.begin_tools("checking", [tool_call()], thinking="reasoning")
    return session, start


def test_v3_sdk_shape_pending_then_completed(tmp_path):
    store = ChatStore(tmp_path)
    session, start = session_with_tools()
    session.set_tool_result(start, {"status": "pending", "detail": "unknown outcome"})
    store.save(session, "model")
    restored, model = store.load(session.session_id)
    assert restored.conversation == session.conversation
    assert restored.needs_response and model == "model"
    session.set_tool_result(start, {"status": "success", "stdout": "hello", "truncated": False})
    session.add_message("assistant", "done", thinking="finished reasoning")
    store.save(session, "model")
    data = json.loads((store.directory / f"{session.session_id}.json").read_text())
    assert data["schema_version"] == 3 and "artifacts" not in data
    assert data["messages"] == session.conversation
    assert store.load(session.session_id)[0].conversation == session.conversation


@pytest.mark.parametrize("status", ["not_started", "pending", "success", "rejected", "error",
                                    "nonzero_exit", "timeout", "cancelled"])
def test_each_outcome_roundtrips(tmp_path, status):
    session, start = session_with_tools()
    session.set_tool_result(start, {"status": status, "detail": "details"})
    data = serialize(session, "model", tmp_path, session.updated_at)
    restored, _ = deserialize(data, tmp_path)
    assert restored.conversation == session.conversation


@pytest.mark.parametrize("corruption", [
    "orphan", "missing", "wrong_name", "wrong_id", "bad_status",
])
def test_rejects_broken_call_result_pairs(tmp_path, corruption):
    session, _ = session_with_tools()
    data = serialize(session, "model", tmp_path, session.updated_at)
    if corruption == "orphan":
        del data["messages"][1]
    elif corruption == "missing":
        data["messages"].pop()
    elif corruption == "wrong_name":
        data["messages"][-1]["tool_name"] = "another"
    elif corruption == "wrong_id":
        data["messages"][-1]["tool_call_id"] = "unknown"
    else:
        data["messages"][-1]["content"] = '{"status":"invented"}'
    with pytest.raises(SessionStorageError):
        deserialize(data, tmp_path)


def test_v1_stays_readable_and_next_save_is_v3(tmp_path):
    session = Session()
    session.add_message("user", "old question")
    session.add_message("assistant", "old answer")
    data = serialize(session, "model", tmp_path, session.updated_at)
    data["schema_version"] = 1
    restored, _ = deserialize(data, tmp_path)
    assert restored.conversation == session.conversation
    assert serialize(restored, "model", tmp_path, restored.updated_at)["schema_version"] == 3


@pytest.mark.parametrize("status", ["success", "error", "interrupted", "incomplete"])
def test_v2_calls_and_receipts_import_without_artifact_io(tmp_path, status):
    session = Session()
    data = {
        "schema_version": 2, "session_id": session.session_id,
        "project": {"path": str(tmp_path)}, "created_at": session.created_at,
        "updated_at": session.updated_at, "model": "old",
        "messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "old-call", "name": "list_files", "arguments": {}},
            ]},
            {"role": "tool", "name": "list_files", "tool_call_id": "old-call",
             "status": status, "artifact_id": "art_0123456789abcdef0123456789abcdef",
             "content": json.dumps({"excerpt": "README.md", "truncated": True,
                                   "artifact_id": "art_0123456789abcdef0123456789abcdef"})},
        ],
        "artifacts": [{"id": "art_0123456789abcdef0123456789abcdef",
                       "path": "artifacts/art_0123456789abcdef0123456789abcdef.txt",
                       "media_type": "text/plain", "size_bytes": 9000, "size_chars": 9000}],
    }
    restored, _ = deserialize(data, tmp_path)
    messages = restored.conversation
    assert messages[1]["tool_calls"] == [{"id": "old-call", "function": {
        "name": "list_files", "arguments": {},
    }}]
    assert messages[2]["tool_call_id"] == "old-call"
    result = json.loads(messages[2]["content"])
    assert result["excerpt"] == "README.md"
    assert result["status"] == ("pending" if status in ("interrupted", "incomplete") else status)
    current = serialize(restored, "new", tmp_path, restored.updated_at)
    assert current["schema_version"] == 3
    assert deserialize(current, tmp_path)[0].conversation == messages


def test_session_snapshots_do_not_mutate_saved_calls(tmp_path):
    session, _ = session_with_tools()
    before = session.conversation
    session.messages[1]["tool_calls"][0]["function"]["arguments"]["command"] = "changed"
    assert session.conversation == before
    assert not session.rollback()  # Command activity is never rolled back.


def test_completed_results_cannot_be_overwritten():
    session, start = session_with_tools()
    session.set_tool_result(start, {"status": "success", "stdout": "original"})
    with pytest.raises(ValueError, match="completed"):
        session.set_tool_result(start, {"status": "pending"})
