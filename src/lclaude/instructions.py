"""Load project guidance without coupling filesystem access to session state."""

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


class InstructionLoadError(Exception):
    """A project instruction file could not be read."""


def load_system_prompt(directory: Path) -> str:
    """Read launch-directory instructions once, preferring the uppercase name."""
    for name in ("AGENTS.md", "agents.md"):
        path = directory / name
        try:
            content = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            raise InstructionLoadError(f"Could not read '{path}': {exc}") from exc
        return content if content.strip() else DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT
