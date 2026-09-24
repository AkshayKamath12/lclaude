"""Slash command handler for lclaude."""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from lclaude.session import Session


@dataclass(frozen=True)
class ShowContext:
    """Request application-coordinated context accounting."""


def _handle_context(session: Session) -> ShowContext:
    return ShowContext()


@dataclass(frozen=True)
class SelectModel:
    """Request application-coordinated model selection."""

    name: str | None = None


def _handle_model(session: Session) -> SelectModel:
    return SelectModel()


def _handle_clear(session: Session) -> bool:
    session.clear()
    sys.stdout.write("\nCleared conversation history.\n")
    return True


def _handle_history(session: Session) -> bool:
    previews = session.get_preview()
    if not previews:
        sys.stdout.write("\n[History is currently empty.]\n")
    else:
        sys.stdout.write(f"\n[Active Context: {len(previews)} messages]\n")
        for idx, role, snippet in previews:
            sys.stdout.write(f"  {idx}. [{role}]: {snippet}...\n")
    return True


def _handle_help(session: Session) -> bool:
    sys.stdout.write("\nAvailable Commands:\n")
    for name, info in COMMANDS.items():
        sys.stdout.write(f"  {name:<10} {info['desc']}\n")
    return True

def _handle_exit(session: Session) -> bool:
    sys.stdout.write("\nExiting session.\n")
    sys.exit(0)

class Command(TypedDict):
    desc: str
    handler: Callable[[Session], bool | SelectModel | ShowContext]


COMMANDS: dict[str, Command] = {
    "/context": {"desc": "Show prompt budget and token usage", "handler": _handle_context},
    "/model": {
        "desc": "Select a model or /model <name>",
        "handler": _handle_model,
    },
    "/clear": {
        "desc": "Clear conversation context history",
        "handler": _handle_clear,
    },
    "/history": {
        "desc": "Display active turn count and roles",
        "handler": _handle_history,
    },
    "/help": {
        "desc": "Show this help message",
        "handler": _handle_help,
    },
    "/exit": {
        "desc": "Terminate the program",
        "handler": _handle_exit,
    },
}

def handle_slash_command(cmd: str, session: Session) -> bool | SelectModel | ShowContext:
    """Main point of entry for slash command handling"""
    command = cmd.strip().lower()

    parts = cmd.strip().split()
    if parts and parts[0].lower() == "/model" and len(parts) > 1:
        if len(parts) == 2:
            return SelectModel(parts[1])
        sys.stdout.write("\nUsage: /model [name]\n")
        return True

    if command in COMMANDS:
        return COMMANDS[command]["handler"](session)

    sys.stdout.write(f"\n[Unknown command: '{cmd}'. Type /help for options.]\n")
    return True

