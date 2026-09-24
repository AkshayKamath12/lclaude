"""Command Line Interface (REPL) for lclaude."""

import argparse
import signal
import sys

from lclaude import ui
from lclaude.commands import COMMANDS, SelectModel, handle_slash_command
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
)
from lclaude.session import Session


def run_chat_loop(engine: InferenceEngine, models: list[str]) -> None:
    """Executes the interactive Read-Eval-Print Loop (REPL)."""
    session = Session()

    commands = {name: info["desc"] for name, info in COMMANDS.items()}
    ui.print_banner(engine.model, engine.host, commands)
    reader = ui.InputReader(commands=commands, models=models)

    while True:
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
                        ui.print_model_changed(engine.model)
                except KeyboardInterrupt:
                    ui.print_model_selection_cancelled()
                except OllamaEngineError as exc:
                    ui.print_error("Model selection failed", str(exc))
            continue

        session.add_message('user', user_input)

        try:
            full_response = ui.render_stream(engine.stream_chat(session.messages))
            session.add_message("assistant", full_response)
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
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    engine = InferenceEngine(
        model=args.model if args.model is not None else "qwen2.5:7b-instruct",
        host = args.host,
        timeout = args.timeout
    )

    try:
        models = engine.verify_ready(allow_fallback=args.model is None)
    except (OllamaConnectionError, ModelNotFoundError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected startup failure: {exc}\n")
        sys.exit(1)

    run_chat_loop(engine, models)

if __name__ == "__main__":
    main()
