"""Budget boundaries and the real application/transport integration."""

from math import ceil
from unittest.mock import patch

import httpx
import ollama
import pytest

from lclaude import ui
from lclaude.cli import parse_args, run_chat_loop
from lclaude.commands import ShowContext, handle_slash_command
from lclaude.context import ConservativeTokenCounter, ContextBudget, PromptCount
from lclaude.engine import InferenceEngine, OllamaEngineError, Usage
from lclaude.instructions import load_system_prompt
from lclaude.session import Session


@pytest.mark.parametrize("instructions", [None, "Use pytest. \u65e5\u672c\u8a9e"])
def test_counts_loaded_or_default_instructions(tmp_path, instructions):
    if instructions is not None:
        (tmp_path / "AGENTS.md").write_text(instructions, encoding="utf-8")
    session = Session(load_system_prompt(tmp_path))
    session.add_message("user", "hi")
    count = ConservativeTokenCounter().count(session.messages)
    assert count.instructions == ceil(len(session.system_prompt.encode("utf-8")) / 3)
    assert count.conversation == 1
    assert count.overhead == 48
    assert count.total == count.instructions + 49
    assert count.estimated


def test_unicode_code_and_empty_message_overhead():
    counter = ConservativeTokenCounter()
    messages = [{"role": "user", "content": "\u00e9\U0001f600\nif x: pass"},
                {"role": "assistant", "content": ""}]
    count = counter.count(messages)
    assert count.conversation == ceil(len(messages[0]["content"].encode("utf-8")) / 3)
    assert count.overhead == 48
    assert counter.count([]).total == 16


@pytest.mark.parametrize("response", [
    {"models": [{"name": "model:latest", "context_length": 8192}]},
    ollama.ProcessResponse(models=[{"model": "model:latest", "context_length": 8192}]),
])
def test_runtime_discovery_uses_ps_and_alias(response):
    engine = InferenceEngine(model="model")
    with patch.object(engine.ollama_client, "ps", return_value=response), patch.object(
        engine.ollama_client, "show", side_effect=AssertionError("no architectural limit")
    ):
        assert engine.context_limit() == 8192


@pytest.mark.parametrize("value", [None, 0, -1, "8192", True])
def test_unknown_runtime_is_not_invented(value):
    engine = InferenceEngine(model="model")
    with patch.object(engine.ollama_client, "ps", return_value={"models": [
        {"model": "other", "context_length": 65536},
        {"model": "model", "context_length": value},
    ]}):
        assert engine.context_limit() is None


def test_cold_model_load_then_query():
    engine = InferenceEngine(model="model")
    with patch.object(engine.ollama_client, "ps", side_effect=[
        {"models": []}, {"models": [{"model": "model", "context_length": 4096}]},
    ]), patch.object(engine.ollama_client, "generate") as load:
        assert engine.context_limit(load=True) == 4096
        load.assert_called_once_with(model="model", prompt="", stream=False)


def test_explicit_context_and_response_are_sent_and_authoritative():
    engine = InferenceEngine(num_ctx=8192, num_predict=2048)
    with patch.object(engine.ollama_client, "ps") as ps, patch.object(
        engine.ollama_client, "chat", return_value=iter([])
    ) as chat:
        assert engine.context_limit(load=True) == 8192
        list(engine.stream_chat([], options={"temperature": 0.2, "num_ctx": 1}))
        ps.assert_not_called()
        assert chat.call_args.kwargs["options"] == {
            "num_ctx": 8192, "num_predict": 2048, "temperature": 0.2,
        }


def test_final_stream_usage_is_separate_from_text():
    engine = InferenceEngine()
    chunks = [ollama.ChatResponse(message={"role": "assistant", "content": "hello"}),
              ollama.ChatResponse(message={"role": "assistant", "content": ""},
                                  done=True, prompt_eval_count=13, eval_count=2)]
    with patch.object(engine.ollama_client, "chat", return_value=iter(chunks)):
        assert list(engine.stream_chat([])) == ["hello"]
    assert engine.last_usage == Usage(13, 2)


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), httpx.ConnectError("lost")])
def test_failed_stream_does_not_publish_usage(failure):
    engine = InferenceEngine()
    engine.last_usage = Usage(100, 30)

    def broken():
        yield {"message": {"content": "partial"}}
        raise failure

    with patch.object(engine.ollama_client, "chat", return_value=broken()):
        with pytest.raises((KeyboardInterrupt, OllamaEngineError)):
            list(engine.stream_chat([]))
    assert engine.last_usage is None


def run_loop(engine, session, inputs):
    with patch("lclaude.cli.Session", return_value=session), patch(
        "lclaude.ui.InputReader.read", side_effect=[*inputs, None]
    ), patch("signal.signal"):
        run_chat_loop(engine, ["old", "new"])


@pytest.mark.parametrize("oversized", ["instructions", "paste"])
def test_oversize_never_blocks_inference(oversized, capsys):
    session = Session("x" * 60000 if oversized == "instructions" else "rules")
    session.add_message("user", "earlier")
    session.add_message("assistant", "answer")
    before = session.messages
    prompt = "hello" if oversized == "instructions" else "x\n" * 30000
    engine = InferenceEngine(model="old", num_ctx=8192)
    with patch.object(engine.ollama_client, "chat", return_value=iter([
        {"message": {"content": "hello back"}},
    ])) as chat:
        run_loop(engine, session, [prompt])
        chat.assert_called_once()
    assert session.messages == before + [
        {"role": "user", "content": prompt}, {"role": "assistant", "content": "hello back"},
    ]
    assert not capsys.readouterr().err


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), httpx.ConnectError("lost")])
def test_discovery_failure_does_not_block_response(failure):
    engine = InferenceEngine(model="old")
    session = Session("rules")
    with patch.object(engine.ollama_client, "ps", side_effect=failure), patch.object(
        engine.ollama_client, "chat", return_value=iter([{"message": {"content": "hi"}}])
    ) as chat:
        run_loop(engine, session, ["hello"])
        chat.assert_called_once()
    assert session.turn_count == 1


def test_unavailable_limit_is_silent_and_never_loads_or_blocks(capsys):
    engine = InferenceEngine(model="old")
    session = Session("rules")
    with patch.object(engine.ollama_client, "ps", return_value={"models": []}), patch.object(
        engine.ollama_client, "generate"
    ) as load, patch.object(engine.ollama_client, "chat", return_value=iter([
        {"message": {"content": "hello"}},
    ])) as chat:
        run_loop(engine, session, ["hi", "/context"])
        chat.assert_called_once()
        load.assert_not_called()
    output = capsys.readouterr()
    assert "unavailable" not in output.out and "%" not in output.out
    assert not output.err
    assert session.turn_count == 1


def test_model_switch_refreshes_runtime_and_clears_reported_usage(capsys):
    engine = InferenceEngine(model="old")
    session = Session("rules")
    with patch.object(engine.ollama_client, "ps", return_value={"models": [
        {"model": "old", "context_length": 8192},
        {"model": "new", "context_length": 4096},
    ]}), patch.object(engine.ollama_client, "chat", side_effect=lambda **kw: iter([
        {"done": True, "message": {"content": "answer"},
         "prompt_eval_count": 9, "eval_count": 1},
    ])) as chat:
        run_loop(engine, session, ["hi", "/context", "/model new", "/context", "next"])
    output = capsys.readouterr().out
    assert "/ 8,192" in output and "/ 4,096" in output
    assert "Last request (tokens reported by Ollama)" in output
    assert "Instructions   ~2" in output
    assert "Conversation   ~" in output
    assert "summary" not in output.lower() and "ledger" not in output.lower()
    assert [call.kwargs["model"] for call in chat.call_args_list] == ["old", "new"]
    assert "num_ctx" not in chat.call_args.kwargs["options"]
    assert chat.call_args.kwargs["options"]["num_predict"] == 2048
    assert output.count("Last request") == 1  # only the explicitly requested old /context


def test_context_command_and_labels(capsys):
    assert isinstance(handle_slash_command(" /CONTEXT ", Session()), ShowContext)
    ui.print_context(PromptCount(10, 20, 30), 8192, ContextBudget(), detail=True)
    ui.print_usage(Usage(50, 7), PromptCount(10, 20, 30))
    output = capsys.readouterr().out
    assert output == (
        "Current context (tokens)\n"
        "  Instructions   ~10\n"
        "  Conversation   ~20\n"
        "  Formatting     ~30\n"
        "  Total          ~60 / 8,192\n"
        "  Reply limit    2,048\n"
        "\nLast request (tokens reported by Ollama)\n"
        "  Prompt         50 (estimated ~60 before sending)\n"
        "  Reply          7\n"
    )
    ui.print_context(PromptCount(10, 20, 30, False), 8192, ContextBudget())
    assert "~" not in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ["--num-ctx", "0"], ["--max-response-tokens", "-1"],
])
def test_invalid_cli_budgets(args):
    with patch("sys.argv", ["lclaude", *args]), pytest.raises(SystemExit):
        parse_args()


def test_regular_chat_does_not_print_context_or_usage(capsys):
    engine = InferenceEngine(model="old", num_ctx=8192)
    with patch.object(engine.ollama_client, "chat", return_value=iter([
        {"done": True, "message": {"content": "hello"},
         "prompt_eval_count": 1400, "eval_count": 1},
    ])):
        run_loop(engine, Session("rules"), ["hello"])
    output = capsys.readouterr().out
    assert "Assistant: hello" in output
    assert "Prompt ~" not in output
    assert "Last request" not in output


def test_runtime_allocation_is_not_cached():
    engine = InferenceEngine(model="old")
    with patch.object(engine.ollama_client, "ps", side_effect=[
        {"models": [{"model": "old", "context_length": 8192}]},
        {"models": [{"model": "old", "context_length": 4096}]},
    ]):
        assert engine.context_limit() == 8192
        assert engine.context_limit(load=True) == 4096


def test_context_unknown_limit_and_missing_reported_counts(capsys):
    ui.print_context(PromptCount(10, 20, 30), None, ContextBudget(), detail=True)
    ui.print_usage(Usage(), PromptCount(10, 20, 30))
    output = capsys.readouterr().out
    assert "Total          ~60\n" in output
    assert "unavailable" not in output
    assert "Last request" not in output
    ui.print_usage(Usage(eval_count=0), PromptCount(10, 20, 30))
    output = capsys.readouterr().out
    assert "Reply          0" in output
    assert "Prompt" not in output
