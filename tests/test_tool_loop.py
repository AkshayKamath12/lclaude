"""Scripted SDK streams exercise the complete command loop and durable checkpoints."""

import json
from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest

from lclaude import ui
from lclaude.cli import parse_args, run_agent_turn, run_chat_loop
from lclaude.context import ConservativeTokenCounter, ContextBudget
from lclaude.engine import InferenceEngine, OllamaEngineError
from lclaude.persistence import ChatStore, SessionStorageError
from lclaude.session import Session
from lclaude.tools import SHELL, TOOL_SCHEMAS, command_result


def call(command="echo hi", name="run_command", **arguments):
    return {"function": {"name": name, "arguments": {
        "command": command, "shell": SHELL, **arguments,
    }}}


def response(content="", calls=None, thinking=""):
    return iter([ollama.ChatResponse(done=True, message={
        "role": "assistant", "content": content, "tool_calls": calls or [], "thinking": thinking,
    })])


@pytest.fixture
def env(tmp_path):
    store = ChatStore(tmp_path, tmp_path / "chats")
    session = Session("rules")
    session.add_message("user", "inspect")
    engine = InferenceEngine(num_ctx=16000)
    reader = MagicMock(context_text="")
    return store, session, engine, reader


def run(env, responses, *, iterations=10, limit=16000, approval="approved", result=None):
    store, session, engine, reader = env
    with patch.object(engine.ollama_client, "chat", side_effect=responses) as chat, patch(
        "lclaude.ui.approve_command", return_value=approval
    ) as approve, patch(
        "lclaude.cli.run_command",
        return_value=result or command_result("success", "done", stdout="hello", stderr="",
                                             exit_code=0, truncated=False),
    ) as execute:
        run_agent_turn(engine, session, store, reader, limit, ContextBudget(), iterations)
    return chat, approve, execute


def results(session):
    return [json.loads(m["content"]) for m in session.conversation if m["role"] == "tool"]


def test_multiple_calls_in_order_and_identical_bounded_result_in_request(env):
    store, session, _, _ = env
    chat, approve, execute = run(env, [
        response("Checking", [call("echo one"), call("echo two")], "Need evidence."),
        response("Done"),
    ])
    assert execute.call_count == approve.call_count == 2
    assert [c.args[0].command for c in execute.call_args_list] == ["echo one", "echo two"]
    transcript = store.load(session.session_id)[0].conversation
    assert [m["role"] for m in transcript] == ["user", "assistant", "tool", "tool", "assistant"]
    assert transcript[1]["thinking"] == "Need evidence."
    assert chat.call_args.kwargs["messages"][1:] == transcript[:-1]
    assert all(r["status"] == "success" for r in results(session))


@pytest.mark.parametrize("status", ["rejected", "cancelled"])
def test_approval_rejection_and_cancellation_never_execute(env, status):
    responses = [response(calls=[call()])]
    if status == "rejected":
        responses.append(response("Rejected"))
    chat, approve, execute = run(env, responses, approval=status)
    execute.assert_not_called()
    approve.assert_called_once()
    assert results(env[1])[0]["status"] == status
    assert chat.call_count == (2 if status == "rejected" else 1)


@pytest.mark.parametrize("tool_call", [call(name="list_files"), call(timeout_seconds=0)])
def test_dispatch_validation_results_without_approval(env, tool_call):
    _, approve, execute = run(env, [response(calls=[tool_call]), response("Invalid request")])
    approve.assert_not_called()
    execute.assert_not_called()
    assert results(env[1])[0]["status"] == "error"


@pytest.mark.parametrize("status", ["error", "nonzero_exit", "timeout", "cancelled", "success"])
def test_command_outcomes_persist_before_followup(env, status):
    store, session, engine, reader = env
    outcome = command_result(status, "details", stdout="bounded", truncated=True)
    events = []
    def execute(command):
        saved, _ = store.load(session.session_id)
        pending = results(saved)[0]
        assert pending["status"] == "pending"
        assert pending["cwd"] == str(store.project)
        events.append("executed")
        return outcome
    def chat(**kwargs):
        if not events:
            return response(calls=[call()])
        assert results(store.load(session.session_id)[0])[0] == outcome
        return response("done")
    with patch.object(engine.ollama_client, "chat", side_effect=chat), patch(
        "lclaude.ui.approve_command", return_value="approved"
    ), patch("lclaude.cli.run_command", side_effect=execute):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    assert results(store.load(session.session_id)[0])[0] == outcome


def test_iteration_limit(env, capsys):
    chat, _, execute = run(env, [response(calls=[call()]) for _ in range(2)], iterations=2)
    assert chat.call_count == execute.call_count == 2
    assert env[1].needs_response
    assert "Tool loop limit" in capsys.readouterr().err


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), httpx.ConnectError("lost")])
def test_failed_followup_keeps_completed_calls_and_discards_partial_assistant(env, failure):
    def broken():
        yield {"message": {"content": "unfinished", "thinking": "partial reasoning"}}
        raise failure
    with pytest.raises((ui.StreamAbortedError, OllamaEngineError)):
        run(env, [response(calls=[call()]), broken()])
    store, session, _, _ = env
    assert store.load(session.session_id)[0].conversation == session.conversation
    assert results(session)[0]["status"] == "success"
    assert "unfinished" not in json.dumps(session.conversation)


def test_incomplete_stream_never_executes_or_publishes_calls(env):
    _, session, engine, _ = env
    with pytest.raises(OllamaEngineError, match="completion marker"):
        run(env, [iter([{"message": {"tool_calls": [call()], "thinking": "partial"}}])])
    assert engine.last_tool_calls == [] and engine.last_thinking == ""
    assert session.conversation == [{"role": "user", "content": "inspect"}]


def test_stream_accumulates_thinking_content_and_calls_before_approval(env):
    store, session, engine, reader = env
    closed = []
    def stream():
        try:
            yield ollama.ChatResponse(message={"role": "assistant", "thinking": "first "})
            yield ollama.ChatResponse(message={"role": "assistant", "content": "Looking "})
            yield ollama.ChatResponse(message={"role": "assistant", "tool_calls": [call()]})
            yield ollama.ChatResponse(done=True, message={
                "role": "assistant", "content": "now", "thinking": "second",
            })
        finally:
            closed.append(True)
    def approval(command):
        assert closed == [True]
        assert session.conversation[1]["content"] == "Looking now"
        assert session.conversation[1]["thinking"] == "first second"
        return "rejected"
    with patch.object(engine.ollama_client, "chat", return_value=stream()), patch(
        "lclaude.ui.approve_command", side_effect=approval
    ):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 1)


@pytest.mark.parametrize("failing_save,expected_executions,saved_status", [
    (1, 0, None), (2, 0, "not_started"), (3, 1, "pending"),
])
def test_save_failures_stop_execution_and_inference(env, failing_save,
                                                  expected_executions, saved_status):
    store, session, _, _ = env
    original_save = store.save
    saves = 0
    def save(chat, model):
        nonlocal saves
        saves += 1
        if saves == failing_save:
            raise SessionStorageError("disk full")
        original_save(chat, model)
    with patch.object(store, "save", side_effect=save), patch(
        "lclaude.cli.run_command", return_value=command_result("success", "done")
    ) as execute, patch("lclaude.ui.approve_command", return_value="approved"), patch.object(
        env[2].ollama_client, "chat", return_value=response(calls=[call(), call()])
    ) as chat:
        with pytest.raises(SessionStorageError):
            run_agent_turn(env[2], session, store, env[3], 16000, ContextBudget(), 10)
    assert execute.call_count == expected_executions
    assert chat.call_count == 1
    if saved_status:
        saved = store.load(session.session_id)[0]
        assert results(saved)[0]["status"] == saved_status
        assert results(saved)[1]["status"] == "not_started"


def test_process_crash_retains_pending_and_never_reruns_on_resume(env, capsys):
    store, session, engine, reader = env
    with patch.object(engine.ollama_client, "chat", return_value=response(calls=[call()])), patch(
        "lclaude.ui.approve_command", return_value="approved"
    ), patch("lclaude.cli.run_command", side_effect=SystemExit(9)), pytest.raises(SystemExit):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    restored, _ = store.load(session.session_id)
    assert results(restored)[0]["status"] == "pending"
    with patch("lclaude.ui.InputReader.read", return_value=None), patch(
        "signal.signal"
    ), patch.object(engine, "stream_chat") as generate, patch(
        "lclaude.cli.run_command"
    ) as execute:
        run_chat_loop(engine, [engine.model], session=restored, store=store)
    generate.assert_not_called()
    execute.assert_not_called()
    assert "unknown outcome" in capsys.readouterr().out
    restored.add_message("user", "what happened?")
    assert "Never automatically repeat" in restored.messages[-2]["content"]


def test_over_budget_followup_is_sent_and_tool_result_is_preserved(env):
    store, session, engine, reader = env
    initial = ConservativeTokenCounter().count(session.messages, TOOL_SCHEMAS).total
    contexts = []
    reader.set_context.side_effect = lambda count, limit, budget: contexts.append(
        (count, limit, budget)
    )
    chat, _, _ = run(env, [response(calls=[call()]), response("done")],
                     limit=initial + 2048, approval="rejected")
    assert chat.call_count == 2
    assert contexts[1][0].total + contexts[1][2].response_tokens > contexts[1][1]
    assert results(store.load(session.session_id)[0])[0]["status"] == "rejected"


def test_truncation_is_visible_and_does_not_gate_final_answer(env, capsys):
    result = command_result("success", "Output truncated; omitted output was not saved.",
                            stdout="prefix", stderr="", exit_code=0, truncated=True)
    chat, _, _ = run(env, [response(calls=[call()]), response("Answer from available output")],
                     result=result)
    assert chat.call_count == 2
    assert results(env[1])[0] == result
    visible = capsys.readouterr().out
    assert "Answer from available output" in visible and "output truncated" in visible


def test_cli_tool_limit_validation():
    with patch("sys.argv", ["lclaude", "--max-tool-iterations", "0"]), pytest.raises(SystemExit):
        parse_args()


def test_cancellation_stops_remaining_calls_and_keeps_completed_results(env):
    store, session, engine, reader = env
    with patch.object(engine.ollama_client, "chat", return_value=response(
        calls=[call("echo first"), call("echo second"), call("echo third")],
    )) as chat, patch("lclaude.ui.approve_command", return_value="approved"), patch(
        "lclaude.cli.run_command", side_effect=[
            command_result("success", "done"), command_result("cancelled", "partial effects"),
        ],
    ) as execute:
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    assert chat.call_count == 1 and execute.call_count == 2
    assert [r["status"] for r in results(store.load(session.session_id)[0])] == [
        "success", "cancelled", "not_started",
    ]


def test_aborted_render_closes_sdk_stream_and_never_persists_partial_message(env):
    store, session, engine, reader = env
    closed = []
    def chunks():
        try:
            yield {"message": {"content": "partial", "tool_calls": [call()]}}
            raise KeyboardInterrupt()
        finally:
            closed.append(True)
    with patch.object(engine.ollama_client, "chat", return_value=chunks()), patch(
        "lclaude.ui.approve_command"
    ) as approval, pytest.raises(ui.StreamAbortedError):
        run_agent_turn(engine, session, store, reader, 16000, ContextBudget(), 10)
    assert closed == [True]
    approval.assert_not_called()
    assert store.list_sessions() == ([], [])
    assert session.conversation == [{"role": "user", "content": "inspect"}]


def test_unsaved_result_blocks_next_user_request_until_checkpoint_succeeds(env, capsys):
    store, session, engine, _ = env
    session.rollback()
    saves = 0
    save = store.save
    def fail_after_pending(chat, model):
        nonlocal saves
        saves += 1
        if saves >= 3:
            raise SessionStorageError("disk full")
        save(chat, model)
    with patch.object(store, "save", side_effect=fail_after_pending), patch(
        "lclaude.ui.InputReader.read", side_effect=["first", "try again", None],
    ), patch("signal.signal"), patch.object(
        engine.ollama_client, "chat", return_value=response(calls=[call()]),
    ) as chat, patch("lclaude.ui.approve_command", return_value="approved"), patch(
        "lclaude.cli.run_command", return_value=command_result("success", "done"),
    ):
        run_chat_loop(engine, [engine.model], session=session, store=store)
    assert chat.call_count == 1
    assert results(session)[0]["status"] == "success"
    assert results(store.load(session.session_id)[0])[0]["status"] == "pending"
    assert capsys.readouterr().err.count("disk full") == 2
