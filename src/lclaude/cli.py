"""Command Line Interface (REPL) for lclaude."""

import argparse
import signal
import sys
from pathlib import Path

from lclaude import ui
from lclaude.commands import (
    COMMANDS,
    ClearConversation,
    SelectChat,
    SelectModel,
    ShowContext,
    handle_slash_command,
)
from lclaude.context import (
    ConservativeTokenCounter,
    ContextBudget,
    ContextBudgetError,
    PromptCount,
    assemble_messages,
)
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
    Usage,
)
from lclaude.instructions import InstructionLoadError, load_system_prompt
from lclaude.persistence import ChatStore, SessionStorageError
from lclaude.session import Session
from lclaude.tools import TOOL_SCHEMAS, ToolError, command_result, run_command, validate_command


def run_chat_loop(
    engine: InferenceEngine, models: list[str], *, system_prompt: str | None = None,
    session: Session | None = None, store: ChatStore, max_tool_iterations: int = 10,
) -> None:
    """Executes the interactive Read-Eval-Print Loop (REPL)."""
    active_session = session if session is not None else Session(system_prompt=system_prompt)

    launch_prompt = active_session.system_prompt

    def save() -> None:
        if active_session.is_empty:
            return
        try:
            store.save(active_session, engine.model)
        except SessionStorageError:
            pass

    commands = {name: info["desc"] for name, info in COMMANDS.items()}
    ui.print_banner(engine.model, engine.host, commands)
    # Show the resumed conversation once when the chat loop starts.
    if not active_session.is_empty:
        ui.render_conversation(active_session.conversation, chat_id=active_session.session_id)
        if active_session.needs_response:
            ui.print_tool_resume()
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
        count = counter.count(active_session.messages, TOOL_SCHEMAS)
        reader.set_context(count, limit, budget)
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
            action = handle_slash_command(user_input, active_session)
            if isinstance(action, ClearConversation):
                # Start a fresh session identity while retaining project instructions.
                active_session.clear()
                sys.stdout.write("\nCleared conversation history.\n")
            elif isinstance(action, SelectModel):
                try:
                    selected = action.name
                    if selected is None:
                        selected = ui.choose_model(models, engine.model)
                    if selected is not None:
                        engine.set_model(selected, models)
                        last_completed_usage = None
                        ui.print_model_changed(engine.model)
                        if not active_session.is_empty:
                            try:
                                store.save_model(active_session.session_id, engine.model)
                            except SessionStorageError:
                                pass
                except KeyboardInterrupt:
                    ui.print_model_selection_cancelled()
                except OllamaEngineError as exc:
                    ui.print_error("Model selection failed", str(exc))
            elif isinstance(action, SelectChat):
                save()
                try:
                    saved, _ = store.list_sessions()
                    choices = [
                        (chat.session_id,
                         f"{ui.format_chat_date(chat.updated_at)}  │  "
                         f"{chat.get_preview()[0][2]}")
                        for chat, _ in saved
                    ]
                    selected = ui.choose_chat(choices, active_session.session_id)
                    if selected is not None and selected != active_session.session_id:
                        restored, model = store.load(selected)
                        engine.set_model(model, models)
                        restored.system_prompt = launch_prompt
                        active_session = restored
                        last_completed_usage = None
                        ui.render_conversation(
                            active_session.conversation, chat_id=active_session.session_id
                        )
                        if active_session.needs_response:
                            ui.print_tool_resume()
                except (SessionStorageError, OllamaEngineError) as exc:
                    ui.print_error("Chat selection failed", str(exc))
                except KeyboardInterrupt:
                    sys.stdout.write("\nChat selection cancelled.\n")
            elif isinstance(action, ShowContext):
                count = counter.count(active_session.messages, TOOL_SCHEMAS)
                ui.print_context(count, limit, budget, detail=True)
                if last_completed_usage is not None:
                    ui.print_usage(*last_completed_usage)
            if active_session.is_empty:
                last_completed_usage = None
            continue

        # A prior result may still be only in memory after a failed checkpoint.
        # Flush it before accepting another request, while the transcript is saveable.
        if any(m["role"] == "tool" for m in active_session.conversation):
            try:
                store.save(active_session, engine.model)
            except SessionStorageError as exc:
                ui.print_error("Agent turn stopped", str(exc))
                continue
        active_session.add_message("user", user_input)
        try:
            last_completed_usage = run_agent_turn(
                engine, active_session, store, reader, limit, budget, max_tool_iterations,
            )
        except (ui.StreamAbortedError, KeyboardInterrupt):
            active_session.rollback()
            ui.print_aborted()
        except (OllamaConnectionError, OllamaEngineError) as exc:
            active_session.rollback()
            ui.print_error("Connection Error", str(exc))
        except (SessionStorageError, ContextBudgetError) as exc:
            active_session.rollback()
            ui.print_error("Agent turn stopped", str(exc))


def run_agent_turn(
    engine: InferenceEngine, session: Session, store: ChatStore, reader: ui.InputReader,
    limit: int | None, budget: ContextBudget, max_iterations: int,
) -> tuple[Usage, PromptCount] | None:
    """Stream, approve, checkpoint, execute, checkpoint, then ask the model again."""
    for _ in range(max_iterations):
        if limit is None:
            limit = engine.context_limit(load=True)
        messages, count = assemble_messages(session.messages, TOOL_SCHEMAS, limit, budget)
        reader.set_context(count, limit, budget)
        stream = engine.stream_chat(messages, tools=TOOL_SCHEMAS)
        try:
            content = ui.render_stream(stream, context_text=reader.context_text)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        calls = engine.last_tool_calls
        if not calls:
            session.add_message("assistant", content, thinking=engine.last_thinking or None)
            try:
                store.save(session, engine.model)
            except SessionStorageError:
                # Retain ordinary chat retry behavior; command checkpoints are strict.
                if any(m["role"] == "tool" for m in session.conversation):
                    raise
            return (engine.last_usage, count) if engine.last_usage is not None else None

        result_start = session.begin_tools(content, calls, engine.last_thinking)
        store.save(session, engine.model)
        for index, call in enumerate(calls):
            function = call["function"]
            name, arguments = function["name"], function.get("arguments", {})
            try:
                command = validate_command(store.project, name, arguments)
            except ToolError as exc:
                result = command_result("error", str(exc))
            else:
                approval = ui.approve_command(command)
                if approval != "approved":
                    result = command_result(approval, "Command was not executed.")
                else:
                    session.set_tool_result(result_start + index, command_result(
                        "pending", "Approved command may have started; outcome is unknown "
                        "until a completed result is saved. Never automatically rerun it.",
                        command=command.command, shell=command.shell, cwd=str(command.cwd),
                        timeout_seconds=command.timeout_seconds,
                    ))
                    # If this save fails, the subprocess must never start.
                    store.save(session, engine.model)
                    ui.print_tool_activity(name, arguments, "running")
                    result = run_command(command)
            session.set_tool_result(result_start + index, result)
            store.save(session, engine.model)
            ui.print_tool_activity(
                name, arguments, result["status"], result.get("truncated", False),
            )
            if result["status"] == "cancelled":
                return None
    ui.print_error("Tool loop limit", f"Stopped after {max_iterations} model responses. "
                   "Command results are saved; enter a new message to continue.")
    return None


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
    parser.add_argument("--max-tool-iterations", type=int, default=10,
                        help="Maximum tool loop iterations (default: 10)")
    args = parser.parse_args()
    if args.num_ctx is not None and args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")
    if args.max_response_tokens <= 0:
        parser.error("Reply tokens must be positive")
    if args.max_tool_iterations <= 0:
        parser.error("Tool loop limit must be positive")
    return args



def main() -> None:
    args = parse_args()

    store = ChatStore(Path.cwd())
    try:
        system_prompt = load_system_prompt(store.project)
        latest, _ = store.latest_session()
        saved_model = None
        if latest is None:
            session = Session(system_prompt)
        else:
            session, saved_model = latest
            session.system_prompt = system_prompt
    except (InstructionLoadError, SessionStorageError, OSError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)

    engine = InferenceEngine(
        model=args.model if args.model is not None else saved_model or "qwen2.5:7b-instruct",
        host = args.host,
        timeout = args.timeout,
        num_ctx=args.num_ctx,
        num_predict=args.max_response_tokens,
    )

    try:
        models = engine.verify_ready(allow_fallback=args.model is None and saved_model is None)
    except (OllamaConnectionError, ModelNotFoundError) as exc:
        sys.stderr.write(f"Startup check failed: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"Unexpected startup failure: {exc}\n")
        sys.exit(1)

    run_chat_loop(engine, models, session=session, store=store,
                  max_tool_iterations=args.max_tool_iterations)

if __name__ == "__main__":
    main()
