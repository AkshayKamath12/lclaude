"""Session management for lclaude"""

from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Session:
    """Manages conversational state, turn tracking, and history rollbacks."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[dict[str, str]]:
        """Returns messages formatted for Ollama API consumption."""
        payload: list[dict[str, str]] = []
        if self.system_prompt:
            payload.append({"role": "system", "content": self.system_prompt})
        payload.extend(m.to_dict() for m in self._messages)
        return payload

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def turn_count(self) -> int:
        """Returns the number of completed user-assistant round trips."""
        return len(self._messages) // 2

    def add_message(self, role: Literal["assistant", "user"], content: str):
        self._messages.append(Message(role, content=content))

    def rollback(self) -> bool:
        "on generation interrupt, removed last message if from user prompt"
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()
            return True
        return False

    def clear(self) -> None:
        self._messages.clear()

    def get_preview(self, max_chars: int = 60) -> list[tuple[int, str, str]]:
        """Returns (index, role, truncated_content) for history inspection."""
        previews = []
        for idx, msg in enumerate(self._messages, 1):
            clean_content = msg.content.replace("\n", " ").strip()
            snippet = (
                clean_content[:max_chars] + "..."
                if len(clean_content) > max_chars
                else clean_content
            )
            previews.append((idx, msg.role, snippet))
        return previews
