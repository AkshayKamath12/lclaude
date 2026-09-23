"""Command Line Interface (REPL) for lclaude."""

import argparse
import sys
import signal

from lclaude.engine import (
    InferenceEngine,
    OllamaEngineError,
    ModelNotFoundError,
    OllamaConnectionError
)

from lclaude.commands import handle_slash_command
from lclaude.session import Session


def run_chat_loop(engine: InferenceEngine) -> None:
    """Executes the interactive Read-Eval-Print Loop (REPL)."""
    session = Session()

    sys.stdout.write("==================================================\n")
    sys.stdout.write("  lclaude - Local Inference Terminal\n")
    sys.stdout.write(f"  Model:   {engine.model}\n")
    sys.stdout.write(f"  Host:    {engine.host}\n")
    sys.stdout.write("  Commands: /clear, /history, /exit, /help\n")
    sys.stdout.write("  Abort generation: Ctrl+C | Exit: Ctrl+C at prompt\n")
    sys.stdout.write("==================================================\n")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            sys.stdout.write("\n\nSession terminated by user.\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_slash_command(user_input, session)
            continue

        session.add_message('user', user_input)
        sys.stdout.write("\nAssistant: ")
        sys.stdout.flush()

        accumulated_tokens: list[str] = []

        # stream generation
        try:
            for token in engine.stream_chat(session.messages):
                sys.stdout.write(token)
                sys.stdout.flush()
                accumulated_tokens.append(token)

            sys.stdout.write("\n")

            # Finalize assistant response in history on clean completion
            session.add_message('assistant', "".join(accumulated_tokens))

        except KeyboardInterrupt:
            # Traps Ctrl+C DURING active generation
            sys.stdout.write("\n\n[Generation aborted by user]\n")
            session.rollback()

        except OllamaConnectionError as exc:
            sys.stderr.write(f"\n\n[Connection Error]: {exc}\n")
            session.rollback()

        except OllamaEngineError as exc:
            sys.stderr.write(f"\n\n[Engine Error]: {exc}\n")
            session.rollback()


def main() -> None:
    """Parses flags, verifies local engine readiness, and launches the REPL."""
    parser = argparse.ArgumentParser(
        prog='lclaude',
        description="Local-first CLI harness for Ollama"
    )

    parser.add_argument(
        "-m",
        "--model",
        default="qwen2.5:7b-instruct",
        help="Ollama model tag to target (default: qwen2.5:7b-instruct)",
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

    args = parser.parse_args()

    engine = InferenceEngine(
        model=args.model,
        host = args.host,
        timeout = args.timeout
    )

    try:
        engine.verify_ready()
    except (OllamaConnectionError, ModelNotFoundError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected startup failure: {exc}\n")
        sys.exit(1)

    run_chat_loop(engine)

if __name__ == "__main__":
    main()