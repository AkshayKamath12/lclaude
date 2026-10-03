"""Command transcripts in the current Ollama message format."""

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
