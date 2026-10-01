"""Artifacts, schema v2 pairing, migration, and snapshot isolation."""

import json
from copy import deepcopy

import pytest

from lclaude.persistence import ChatStore, SessionStorageError, deserialize, serialize
from lclaude.session import Session


def tool_session(store):
    session = Session()
    session.add_message("user", "inspect")
    artifact = store.save_artifact(session, "complete output")
    start = session.begin_tools("Looking.", [
        {"id": "call-a", "name": "read_file", "arguments": {"path": "a.py"}},
        {"id": "call-b", "name": "list_files", "arguments": {}},
    ])
    session.complete_tool(start, "excerpt", "success", artifact)
    return session, artifact


def test_artifact_and_interrupted_tool_turn_round_trip(tmp_path):
    store = ChatStore(tmp_path)
    session, artifact = tool_session(store)
    store.save(session, "model")
    restored, _ = store.load(session.session_id)
    assert restored.conversation == session.conversation
    assert restored.needs_response
    assert restored.turn_count == 0
    assert store.read_artifact(restored, artifact) == ("complete output", 15)
    assert store.read_artifact(restored, artifact, 9, 3) == ("out", 15)
    path = store.directory / session.session_id / session.artifacts[0]["path"]
    assert path.read_text(encoding="utf-8") == "complete output"
    restored.add_message("user", "continue using saved results")
    restored.add_message("assistant", "done")
    store.save(restored, "model")
    assert store.load(restored.session_id)[0].turn_count == 1


def test_v1_migrates_only_on_save(tmp_path):
    store = ChatStore(tmp_path)
    session = Session()
    session.add_message("user", "q")
    session.add_message("assistant", "a")
    data = serialize(session, "m", store.project, session.updated_at)
    data["schema_version"] = 1
    del data["artifacts"]
    store.directory.mkdir(parents=True)
    path = store.directory / f"{session.session_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()
    restored, model = store.load(session.session_id)
    assert path.read_bytes() == original
    assert model == "m" and restored.turn_count == 1
    store.save(restored, model)
    assert json.loads(path.read_text())["schema_version"] == 2


@pytest.mark.parametrize("mutation", ["order", "missing", "name", "duplicate",
                                     "artifact", "path", "status"])
def test_invalid_v2_rejected(tmp_path, mutation):
    store = ChatStore(tmp_path)
    session, artifact = tool_session(store)
    data = serialize(session, "m", store.project, session.updated_at)
    if mutation == "order":
        data["messages"][2], data["messages"][3] = data["messages"][3], data["messages"][2]
    elif mutation == "missing":
        data["messages"].pop()
    elif mutation == "name":
        data["messages"][2]["name"] = "other"
    elif mutation == "duplicate":
        data["messages"][1]["tool_calls"][1]["id"] = "call-a"
    elif mutation == "artifact":
        data["messages"][2]["artifact_id"] = "missing"
    elif mutation == "path":
        data["artifacts"][0]["path"] = "../outside.txt"
    else:
        data["messages"][2]["status"] = "invented"
    with pytest.raises(SessionStorageError):
        deserialize(data, store.project)


def test_missing_cross_session_artifact_and_invalid_ranges(tmp_path):
    store = ChatStore(tmp_path)
    session, artifact = tool_session(store)
    for other, start, size in [(Session(), 0, 1), (session, -1, 1),
                                (session, 100, 1), (session, 0, True)]:
        with pytest.raises(SessionStorageError):
            store.read_artifact(other, artifact, start, size)
    target = store.directory / session.session_id / session.artifacts[0]["path"]
    target.unlink()
    with pytest.raises(SessionStorageError):
        store.read_artifact(session, artifact)
    with pytest.raises(SessionStorageError):
        store.save(session, "model")


def test_session_snapshots_and_clear_do_not_alias_calls(tmp_path):
    store = ChatStore(tmp_path)
    session, _ = tool_session(store)
    original = deepcopy(session.conversation)
    snapshot = session.conversation
    snapshot[1]["tool_calls"][0]["arguments"]["path"] = "changed"
    assert session.conversation == original
    session.clear()
    assert session.artifacts == [] and session.is_empty
