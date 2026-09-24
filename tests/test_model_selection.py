"""Model selection across command, provider, terminal, and application boundaries."""

from unittest.mock import patch

import httpx
import ollama
import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lclaude import ui
from lclaude.cli import run_chat_loop
from lclaude.commands import SelectModel, handle_slash_command
from lclaude.engine import InferenceEngine, ModelNotFoundError, OllamaEngineError
from lclaude.session import Session


@pytest.mark.parametrize("text, name", [
    ("/model", None), ("  /MODEL  ", None), ("/model MyModel:7B", "MyModel:7B"),
])
def test_parse_model_request(text, name):
    session = Session()
    assert handle_slash_command(text, session) == SelectModel(name)
    assert session.messages == []


def test_extra_arguments_are_rejected(capsys):
    assert handle_slash_command("/model one two", Session()) is True
    assert "Usage:" in capsys.readouterr().out


def test_list_normalizes_sdk_and_legacy_names_without_mutation():
    engine = InferenceEngine(model="old")
    for response in (
        {"models": [{"name": "b"}, {"model": "a"}, {"model": "a"}, {}]},
        ollama.ListResponse(models=[{"model": "b"}, {"model": "a"}, {"model": "a"}]),
    ):
        with patch.object(engine.ollama_client, "list", return_value=response):
            assert engine.list_models() == ["a", "b"]
            assert engine.model == "old"


def test_switch_resolves_latest_and_next_request_uses_selection():
    engine = InferenceEngine(model="old")
    with (
        patch.object(engine.ollama_client, "list", return_value={
            "models": [{"model": "new:latest"}],
        }),
        patch.object(engine.ollama_client, "chat", return_value=iter([])) as chat,
    ):
        engine.set_model("new", ["new:latest"])
        assert engine.model == "new:latest"
        list(engine.stream_chat([]))
        assert chat.call_args.kwargs["model"] == "new:latest"


def test_startup_catalog_is_reused_for_multiple_selections():
    engine = InferenceEngine(model="old")
    with patch.object(engine.ollama_client, "list", return_value={
        "models": [{"model": "old"}, {"model": "new:latest"}],
    }) as listing:
        models = engine.verify_ready()
        engine.set_model("new", models)
        engine.set_model("old", models)
        with pytest.raises(ModelNotFoundError):
            engine.set_model("installed-later", models)
        assert engine.model == "old"
        listing.assert_called_once()


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("offline"), httpx.ReadTimeout("slow"),
    ollama.ResponseError("unavailable", status_code=500), KeyboardInterrupt(),
])
@pytest.mark.parametrize("allow_fallback", [False, True])
def test_failed_startup_preserves_previous_model(failure, allow_fallback):
    engine = InferenceEngine(model="old")
    expected = KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else OllamaEngineError
    with patch.object(engine.ollama_client, "list", side_effect=failure):
        with pytest.raises(expected):
            engine.verify_ready(allow_fallback=allow_fallback)
    assert engine.model == "old"


def test_missing_selection_and_listing_404_have_accurate_errors():
    engine = InferenceEngine(model="old")
    with patch.object(engine.ollama_client, "list", return_value={"models": []}):
        with pytest.raises(ModelNotFoundError, match="'new'"):
            engine.set_model("new", [])
    with patch.object(engine.ollama_client, "list", side_effect=ollama.ResponseError(
        "endpoint missing", status_code=404,
    )):
        with pytest.raises(OllamaEngineError, match="listing models") as error:
            engine.list_models()
        assert not isinstance(error.value, ModelNotFoundError)
    assert engine.model == "old"


@pytest.mark.parametrize("keys, expected", [
    ("\r", "b"), ("\x1b[A\r", "a"), ("\x1b[B\r", "a"),
    ("\x1b", None), ("\x03", None), ("\x04", None), ("\x1a", None),
])
def test_picker_real_key_processing(keys, expected):
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        assert ui.choose_model(
            ["a", "b"], "b", input_stream=pipe, output_stream=DummyOutput(),
        ) == expected


def test_noninteractive_picker_lists_without_consuming_input(capsys):
    with patch("builtins.input", side_effect=AssertionError("must not read")):
        assert ui.choose_model(["a", "b"], "b") is None
    output = capsys.readouterr().out
    assert "/model <name>" in output
    assert "b (current)" in output


def test_empty_picker_does_not_open_terminal(capsys):
    with patch("lclaude.ui.Application", side_effect=AssertionError("must not open")):
        assert ui.choose_model([], "old") is None
    assert "No models available" in capsys.readouterr().err


@pytest.mark.parametrize("command, selection, failure", [
    ("/model", "new", None), ("/model new", None, None),
    ("/model", None, None), ("/model missing", None, ModelNotFoundError("missing")),
    ("/model", "new", KeyboardInterrupt()),
])
def test_loop_preserves_history_and_continues(command, selection, failure):
    engine = InferenceEngine(model="old")
    session = Session()
    session.add_message("user", "earlier")
    session.add_message("assistant", "answer")
    baseline = session.messages
    with (
        patch("lclaude.cli.Session", return_value=session),
        patch("lclaude.ui.InputReader.read", side_effect=[command, "next", None]),
        patch("lclaude.ui.choose_model", return_value=selection) as picker,
        patch.object(engine.ollama_client, "list", side_effect=AssertionError("no refetch")),
        patch.object(engine, "set_model", wraps=engine.set_model, side_effect=failure),
        patch.object(engine.ollama_client, "chat", return_value=iter([
            {"message": {"content": "done"}},
        ])) as chat,
        patch("signal.signal"),
    ):
        run_chat_loop(engine, ["old", "new"])
    switched = failure is None and (selection is not None or command == "/model new")
    assert engine.model == ("new" if switched else "old")
    assert chat.call_args.kwargs["model"] == engine.model
    assert chat.call_args.kwargs["messages"] == baseline + [{"role": "user", "content": "next"}]
    assert session.messages == baseline + [
        {"role": "user", "content": "next"}, {"role": "assistant", "content": "done"},
    ]
    if command == "/model new":
        picker.assert_not_called()
