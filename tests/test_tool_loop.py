"""Real application/engine/session integration with scripted Ollama streams."""

import json
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from lclaude import ui
from lclaude.cli import parse_args, run_agent_turn, run_chat_loop
from lclaude.context import (
    ConservativeTokenCounter,
    ContextBudget,
    assemble_messages,
    pending_artifacts,
)
from lclaude.engine import InferenceEngine, OllamaEngineError
from lclaude.persistence import ChatStore, SessionStorageError
from lclaude.session import Session
from lclaude.tools import TOOL_SCHEMAS


def call(name, args, call_id=None):
    item = {"function": {"name": name, "arguments": args}}
    if call_id is not None:
        item["id"] = call_id
    return item


def response(content="", calls=None):
    return iter([{"done": True, "message": {"content": content, "tool_calls": calls or []}}])


@pytest.fixture
def env(tmp_path):
    store = ChatStore(tmp_path)
    session = Session()
    session.add_message("user", "inspect")
    engine = InferenceEngine(num_ctx=16000)
    reader = MagicMock(spec=ui.InputReader)
    reader.context_text = ""
    return store, session, engine, reader


def run(env, responses, iterations=10):
    store, session, engine, reader = env
    with patch.object(engine.ollama_client, "chat", side_effect=responses) as chat:
        usage = run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), iterations)
    return chat, usage


def test_multiple_calls_keep_one_assistant_message_and_matching_order(env):
    store, session, _, _ = env
    (store.project / "a.txt").write_text("hello", encoding="utf-8")
    chat, _ = run(env, [
        response("I will inspect.", [
            call("read_file", {"path": "a.txt"}, "provided-id"),
            call("list_files", {}),
        ]),
        response("done"),
    ])
    assert chat.call_count == 2
    transcript = session.conversation
    assert [m["role"] for m in transcript] == ["user", "assistant", "tool", "tool", "assistant"]
    assert transcript[1]["content"] == "I will inspect."
    calls = transcript[1]["tool_calls"]
    assert calls[0]["id"] == "provided-id"
    assert calls[1]["id"]
    assert [m["tool_call_id"] for m in transcript[2:4]] == [c["id"] for c in calls]
    assert store.load(session.session_id)[0].conversation == transcript
    request = chat.call_args.kwargs
    assert request["tools"] == TOOL_SCHEMAS
    assert request["messages"][2]["tool_name"] == "read_file"
    assert session.turn_count == 1


def test_sdk_tool_calls_and_empty_content(env):
    store, session, engine, reader = env
    sdk_response = ollama.ChatResponse(done=True, message={
        "role": "assistant", "tool_calls": [
            {"function": {"name": "list_files", "arguments": {}}},
        ],
    })
    with patch.object(engine.ollama_client, "chat", side_effect=[
        iter([sdk_response]), response("done"),
    ]):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    assert session.conversation[2]["status"] == "success"


def test_full_output_saved_before_excerpt_and_retrieval(env):
    store, session, engine, reader = env
    full = "long line\n" * 10
    (store.project / "a.txt").write_text(full, encoding="utf-8")
    requests = []

    def chat(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return response(calls=[call("read_file", {"path": "a.txt", "max_chars": 20})])
        if len(requests) == 2:
            saved, _ = store.load(session.session_id)
            receipt = json.loads(saved.conversation[-1]["content"])
            assert receipt["truncated"] and len(receipt["excerpt"]) == 20
            assert store.read_artifact(saved, receipt["artifact_id"], 0, 12000)[0] == full
            return response(calls=[call("read_tool_output", {
                "artifact_id": receipt["artifact_id"], "start": 0, "max_chars": 30,
            })])
        if len(requests) == 3:
            return response(calls=[call("read_tool_output", {
                "artifact_id": session.artifacts[0]["id"], "start": 30, "max_chars": 12000,
            })])
        return response("done")

    with patch.object(engine.ollama_client, "chat", side_effect=chat):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    results = [json.loads(m["content"]) for m in session.conversation if m["role"] == "tool"]
    assert results[1]["excerpt"] == full[:30]
    assert results[1]["artifact_id"] == results[0]["artifact_id"]
    assert results[1]["next_start"] == 30 and results[1]["truncated"]
    assert results[2]["excerpt"] == full[30:]
    assert len(session.artifacts) == 1


@pytest.mark.parametrize("name,args", [
    ("shell", {"command": "bad"}),
    ("read_file", {"path": False}),
    ("read_file", {"path": "missing"}),
    ("read_file", "not an object"),
    ("read_tool_output", {"artifact_id": "not-mine"}),
])
def test_validation_errors_return_to_model_as_paired_results(env, name, args):
    _, session, _, _ = env
    chat, _ = run(env, [response(calls=[call(name, args)]), response("handled")])
    assert session.conversation[2]["status"] == "error"
    assert chat.call_args.kwargs["messages"][2]["content"].startswith("Tool error:")


def test_loop_limit_stops_with_saved_pairs(env, capsys):
    store, session, _, _ = env
    chat, _ = run(env, [
        response(calls=[call("list_files", {})]),
        response(calls=[call("list_files", {})]),
    ], iterations=2)
    assert chat.call_count == 2
    assert session.needs_response and store.load(session.session_id)[0].needs_response
    assert "Tool loop limit" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), httpx.ConnectError("lost")])
def test_interrupted_following_inference_keeps_completed_pair(env, failure):
    store, session, _, _ = env

    def broken():
        yield {"message": {"content": "unfinished"}}
        raise failure

    with pytest.raises((ui.StreamAbortedError, OllamaEngineError)):
        run(env, [response(calls=[call("list_files", {})]), broken()])
    restored, _ = store.load(session.session_id)
    assert restored.conversation == session.conversation
    assert restored.conversation[-1]["status"] == "success"
    assert "unfinished" not in json.dumps(restored.conversation)


def test_interruption_between_calls_preserves_original_bundle_and_unexecuted_status(env):
    from lclaude.tools import dispatch

    store, session, _, _ = env
    attempts = []

    def interrupted(root, name, arguments, reader):
        attempts.append(name)
        if len(attempts) == 2:
            raise KeyboardInterrupt()
        return dispatch(root, name, arguments, reader)

    with patch("lclaude.cli.dispatch", side_effect=interrupted), pytest.raises(KeyboardInterrupt):
        run(env, [response(calls=[call("list_files", {}), call("list_files", {})])])
    restored, _ = store.load(session.session_id)
    assert [m["status"] for m in restored.conversation[2:]] == ["success", "interrupted"]
    assert len(restored.conversation[1]["tool_calls"]) == 2


def test_failed_artifact_write_stops_before_next_call(env):
    store, session, _, _ = env
    with patch.object(store, "save_artifact", side_effect=SessionStorageError("disk full")):
        with pytest.raises(SessionStorageError):
            run(env, [response(calls=[call("list_files", {}), call("list_files", {})])])
    restored, _ = store.load(session.session_id)
    assert all(m["status"] == "interrupted" for m in restored.conversation[2:])
    assert not restored.artifacts


def test_snapshot_failure_after_result_does_not_execute_next_call(env):
    from lclaude.tools import dispatch

    store, session, _, _ = env
    save = store.save
    attempts = 0

    def fail_result(chat, model):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise SessionStorageError("disk full")
        save(chat, model)

    with patch.object(store, "save", side_effect=fail_result), patch(
        "lclaude.cli.dispatch", wraps=dispatch
    ) as execute, pytest.raises(SessionStorageError):
        run(env, [response(calls=[call("list_files", {}), call("list_files", {})])])
    assert execute.call_count == 1
    assert session.conversation[2]["status"] == "success"
    assert store.load(session.session_id)[0].conversation[2]["status"] == "interrupted"


def test_resume_displays_status_and_never_generates_without_input(env, capsys):
    store, session, engine, _ = env
    run(env, [response(calls=[call("list_files", {})])], iterations=1)
    restored, _ = store.load(session.session_id)
    with patch("lclaude.ui.InputReader.read", return_value=None), patch(
        "signal.signal"
    ), patch.object(engine, "stream_chat") as generate:
        run_chat_loop(engine, [engine.model], session=restored, store=store)
    generate.assert_not_called()
    output = capsys.readouterr().out
    assert "no calls are repeated automatically" in output
    assert "list_files" in output and "done" in output


def test_context_counts_tools_arguments_and_retrieval_without_mutating_transcript(env):
    store, session, _, _ = env
    (store.project / "a").write_text("x" * 12000, encoding="utf-8")
    run(env, [response(calls=[call("read_file", {"path": "a"})])], iterations=1)
    original = session.conversation
    counter = ConservativeTokenCounter()
    count = counter.count(session.messages, TOOL_SCHEMAS)
    assert count.tools > 0
    assert count.total > counter.count(session.messages).total
    budget = ContextBudget(100)
    payload, counted = assemble_messages(session.messages, TOOL_SCHEMAS, 1, budget)
    assert counted.total > 1  # Advisory even when the estimate exceeds allocation.
    assert session.conversation == original
    assert [m["role"] for m in payload[:-1]] == [m["role"] for m in session.messages]
    assert "not the output" in payload[-2]["content"]
    assert "x" * 20 not in payload[-2]["content"]
    assert "read_tool_output" in payload[-1]["content"]
    assert counted == counter.count(payload, TOOL_SCHEMAS)


def test_incomplete_stream_never_publishes_calls_or_usage(env):
    _, _, engine, _ = env
    with patch.object(engine.ollama_client, "chat", return_value=iter([
        {"message": {"content": "partial", "tool_calls": [call("list_files", {})]}},
    ])), pytest.raises(OllamaEngineError, match="completion marker"):
        list(engine.stream_chat([]))
    assert engine.last_tool_calls == [] and engine.last_usage is None


def test_tool_activity_escapes_and_bounds_arguments(capsys):
    ui.print_tool_activity("read_file", {"path": "\x1b[2J" + "x" * 1000}, "success", True)
    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert len(output) < 260 and "remaining output saved" in output


def test_cli_tool_limit_validation():
    with patch("sys.argv", ["lclaude", "--max-tool-iterations", "0"]):
        with pytest.raises(SystemExit):
            parse_args()


def test_wire_call_ids_and_invalid_argument_shapes_survive_sdk_boundary():
    engine = InferenceEngine()
    raw = {
        "done": True,
        "message": {"content": "", "tool_calls": [
            call("read_file", {"path": "a"}, "server-provided"),
            call("read_file", ["invalid", "arguments"], "server-invalid"),
        ]},
    }
    wire = httpx.Response(200, text=json.dumps(raw) + "\n")
    with patch("lclaude.engine.httpx.stream") as http_stream:
        http_stream.return_value.__enter__.return_value = wire
        assert list(engine.stream_chat([], tools=TOOL_SCHEMAS)) == []
    assert engine.last_tool_calls == raw["message"]["tool_calls"]
    assert http_stream.call_args.kwargs["json"]["tools"] == TOOL_SCHEMAS


@pytest.mark.parametrize("status,content", [
    (500, "server error"), (200, '{"error":"model failed"}'), (200, "[]"),
])
def test_wire_failures_become_engine_errors(status, content):
    engine = InferenceEngine()
    with patch("lclaude.engine.httpx.stream") as http_stream:
        http_stream.return_value.__enter__.return_value = httpx.Response(status, text=content)
        with pytest.raises(OllamaEngineError):
            list(engine.stream_chat([]))
    assert engine.last_tool_calls == [] and engine.last_usage is None


def test_stream_is_closed_when_rendering_is_interrupted(env):
    store, session, engine, reader = env
    closed = []

    def chunks():
        try:
            yield {"message": {"content": "partial"}}
            raise KeyboardInterrupt()
        finally:
            closed.append(True)

    with patch.object(engine.ollama_client, "chat", return_value=chunks()):
        with pytest.raises(ui.StreamAbortedError):
            run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    assert closed == [True]
    assert session.conversation == [{"role": "user", "content": "inspect"}]


def test_supplied_absolute_path_must_stay_inside_project(tmp_path):
    from lclaude.tools import ToolError, dispatch

    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ToolError, match="outside"):
        dispatch(project, "read_file", {"path": str(outside.resolve())})


def test_small_context_never_blocks_tool_followup_or_shrinks_names(env, capsys):
    store, session, engine, reader = env
    session.system_prompt = "Project instructions. " * 1000
    (store.project / "CONTRIBUTING.md").write_text("notes", encoding="utf-8")
    with patch.object(engine.ollama_client, "chat", side_effect=[
        response(calls=[call("list_files", {})]),
        response("CONTRIBUTING.md"),
    ]) as chat:
        run_agent_turn(engine, session, store, reader, 4096, ContextBudget(2048), 10)
    assert chat.call_count == 2
    tool = next(m for m in chat.call_args.kwargs["messages"] if m["role"] == "tool")
    assert "CONTRIBUTING.md\n" in tool["content"]
    output = capsys.readouterr()
    assert "CONTRIBUTING.md" in output.out
    assert not output.err


def test_premature_preview_answers_are_hidden_until_all_artifact_ranges_read(env, capsys):
    store, session, engine, reader = env
    (store.project / "a.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    step = 0

    def chat(**kwargs):
        nonlocal step
        step += 1
        if step == 1:
            return response(calls=[call("read_file", {"path": "a.txt", "max_chars": 2})])
        if step in (2, 4):
            return response('alp... Would you like the full output? {"artifact_id": "receipt"}')
        if step in (3, 5):
            artifact_id = session.artifacts[0]["id"]
            start = 0 if step == 3 else 6
            assert any("read_tool_output" in m["content"] for m in kwargs["messages"])
            return response(calls=[call("read_tool_output", {
                "artifact_id": artifact_id, "start": start, "max_chars": 6 if step == 3 else 12000,
            })])
        assert step == 6
        return response("alpha\nbeta\ngamma")

    with patch.object(engine.ollama_client, "chat", side_effect=chat):
        run_agent_turn(engine, session, store, reader, 4096, ContextBudget(), 10)
    visible = capsys.readouterr().out
    assert "alpha\nbeta\ngamma" in visible
    assert "alp..." not in visible and "Would you like" not in visible
    assert "artifact_id" not in visible and "receipt" not in visible
    assert "Would you like" not in json.dumps(session.conversation)
    assert not pending_artifacts(session.conversation)


def test_retrieval_gate_remains_bounded_if_model_ignores_it(env, capsys):
    store, session, _, _ = env
    (store.project / "a").write_text("complete data", encoding="utf-8")
    chat, _ = run(env, [
        response(calls=[call("read_file", {"path": "a", "max_chars": 1})]),
        response("I will answer from the receipt."),
        response("I will answer from the receipt again."),
    ], iterations=3)
    output = capsys.readouterr()
    assert chat.call_count == 3
    assert "answer from the receipt" not in output.out
    assert "Tool loop limit" in output.err
    assert session.needs_response


def test_readable_activity_for_live_and_resumed_calls(capsys):
    ui.print_tool_activity("list_files", {"glob": "*", "limit": 100}, "running")
    ui.print_tool_activity("list_files", {"glob": "*", "limit": 100}, "success")
    ui.print_tool_activity(
        "read_tool_output", {"artifact_id": "secret-id", "start": 100}, "success"
    )
    visible = capsys.readouterr().out
    assert "list_files: . (pattern *) - running" in visible
    assert "list_files: . (pattern *) - done" in visible
    assert "saved output from character 100" in visible
    assert all(raw not in visible for raw in ('[Tool', '{"', 'artifact_id', 'secret-id'))


def test_resumed_pending_artifact_stays_pending_and_gaps_are_not_skipped(env):
    store, session, _, _ = env
    (store.project / "a").write_text("abcdefghijkl", encoding="utf-8")
    run(env, [response(calls=[call("read_file", {"path": "a", "max_chars": 2})])], iterations=1)
    artifact_id = session.artifacts[0]["id"]
    restored, _ = store.load(session.session_id)
    restored.add_message("user", "show all of it")
    assert pending_artifacts(restored.conversation) == {artifact_id: 0}
    start = restored.begin_tools("", [{
        "id": "later-range", "name": "read_tool_output",
        "arguments": {"artifact_id": artifact_id, "start": 4},
    }])
    restored.complete_tool(start, json.dumps({
        "artifact_id": artifact_id, "start": 4, "next_start": 12, "total_chars": 12,
        "excerpt": "efghijkl", "truncated": False,
    }), "success", artifact_id)
    assert pending_artifacts(restored.conversation) == {artifact_id: 0}
