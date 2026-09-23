"""Slash command handler for lclaude."""

import sys
from typing import Any, Callable

from lclaude.session import Session

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

COMMANDS: dict[str, dict[str, Any]] = {
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

def handle_slash_command(cmd: str, session: Session) -> bool:
    """Main point of entry for slash command handling"""
    command = cmd.strip().lower()

    if command in COMMANDS:
        return COMMANDS[command]["handler"](session)

    sys.stdout.write(f"\n[Unknown command: '{cmd}'. Type /help for options.]\n")
    return True

