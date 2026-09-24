"""Command Line Interface (REPL) for lclaude."""

import argparse
import signal
import sys
from pathlib import Path

from lclaude import ui
from lclaude.commands import COMMANDS, SelectModel, ShowContext, handle_slash_command
from lclaude.context import ConservativeTokenCounter, ContextBudget
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
)
from lclaude.instructions import InstructionLoadError, load_system_prompt
from lclaude.session import Session


def run_chat_loop(
    engine: InferenceEngine, models: list[str], *, system_prompt: str | None = None
) -> None:
    """Executes the interactive Read-Eval-Print Loop (REPL)."""
    session = Session(system_prompt=system_prompt)

    commands = {name: info["desc"] for name, info in COMMANDS.items()}
    ui.print_banner(engine.model, engine.host, commands)
    reader = ui.InputReader(commands=commands, models=models)

    counter = ConservativeTokenCounter()
    budget = ContextBudget(engine.num_predict)
    last_completed_usage = None

    while True:
        try:
            limit = engine.context_limit()
        except OllamaEngineError:
            limit = None
        except KeyboardInterrupt:
            ui.print_aborted()
            limit = None
        reader.set_context(counter.count(session.messages), limit, budget)
        user_input = reader.read()

        if user_input is None:
            # Permanently ignore SIGINT here so a user pressing 
            # Ctrl+C a second time doesn't trigger an unhandled traceback.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            ui.print_session_end()
            break

        if not user_input.strip():
            continue

        if user_input.lstrip().startswith("/"):
            action = handle_slash_command(user_input, session)
            if isinstance(action, SelectModel):
                try:
                    selected = action.name
                    if selected is None:
                        selected = ui.choose_model(models, engine.model)
                    if selected is not None:
                        engine.set_model(selected, models)
                        last_completed_usage = None
                        ui.print_model_changed(engine.model)
                except KeyboardInterrupt:
                    ui.print_model_selection_cancelled()
                except OllamaEngineError as exc:
                    ui.print_error("Model selection failed", str(exc))
            elif isinstance(action, ShowContext):
                ui.print_context(counter.count(session.messages), limit, budget, detail=True)
                if last_completed_usage is not None:
                    ui.print_usage(*last_completed_usage)
            if session.is_empty:
                last_completed_usage = None
            continue

        session.add_message("user", user_input)
        count = counter.count(session.messages)
        reader.set_context(count, limit, budget)
        try:
            full_response = ui.render_stream(
                engine.stream_chat(session.messages), context_text=reader.context_text
            )
            session.add_message("assistant", full_response)
            if engine.last_usage is not None:
                last_completed_usage = (engine.last_usage, count)
            else:
                last_completed_usage = None
        except ui.StreamAbortedError:
            session.rollback()
            ui.print_aborted()
        except (OllamaConnectionError, OllamaEngineError) as exc:
            session.rollback()
            ui.print_error("Connection Error", str(exc))

def parse_args() -> argparse.Namespace:
    """Parses flags, verifies local engine readiness, and launches the REPL."""
    parser = argparse.ArgumentParser(
        prog='lclaude',
        description="Local-first CLI harness for Ollama"
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Ollama model tag (default: qwen2.5:7b-instruct, or first installed model)",
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama daemon endpoint (default: http://localhost:11434)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Client socket timeout in seconds (default: 60.0)",
    )
    parser.add_argument("--num-ctx", type=int, default=None,
                        help="Explicit Ollama context allocation; otherwise discover runtime size")
    parser.add_argument("--max-response-tokens", type=int, default=2048,
                        help="Maximum reply tokens (default: 2048)")
    args = parser.parse_args()
    if args.num_ctx is not None and args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")
    if args.max_response_tokens <= 0:
        parser.error("Reply tokens must be positive")
    return args



def main() -> None:
    args = parse_args()

    try:
        system_prompt = load_system_prompt(Path.cwd())
    except (InstructionLoadError, OSError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)

    engine = InferenceEngine(
        model=args.model if args.model is not None else "qwen2.5:7b-instruct",
        host = args.host,
        timeout = args.timeout,
        num_ctx=args.num_ctx,
        num_predict=args.max_response_tokens
    )

    try:
        models = engine.verify_ready(allow_fallback=args.model is None)
    except (OllamaConnectionError, ModelNotFoundError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected startup failure: {exc}\n")
        sys.exit(1)

    run_chat_loop(engine, models, system_prompt=system_prompt)

if __name__ == "__main__":
    main()
