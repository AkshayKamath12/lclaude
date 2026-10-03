"""Ordered conversation state and command checkpoints."""

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_name: str | None = None
    thinking: str | None = None
    tool_call_id: str | None = None  # ID of the assistant tool call this result answers.

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        for key in ("tool_calls", "tool_name", "thinking", "tool_call_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return deepcopy(result)


class Session:
    """Own the original transcript; inference receives snapshots of it."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt
        self._new_identity()
        self._messages: list[Message] = []

    def _new_identity(self) -> None:
        """Refresh identity and timestamps so the next save records a new chat."""
        self.session_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

    @property
    def conversation(self) -> list[dict[str, Any]]:
        """Return the full transcript, excluding the separately stored system prompt."""
        return [message.to_dict() for message in self._messages]

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return messages formatted for Ollama, including resume guidance when needed."""
        payload: list[dict[str, Any]] = []
        if self.system_prompt:
            payload.append({"role": "system", "content": self.system_prompt})
        for item in self.conversation:
            if item["role"] == "user" and payload and payload[-1]["role"] == "tool":
                payload.append({"role": "system", "content":
                    "The previous turn stopped after tool work. Never automatically repeat "
                    "recorded commands. A pending result means execution may have happened "
                    "but its outcome is unknown. Follow the user's next request."})
            payload.append(item)
        return payload

    @property
    def is_empty(self) -> bool:
        return not self._messages

    @property
    def turn_count(self) -> int:
        """Count completed assistant responses, excluding tool-call requests."""
        return sum(m.role == "assistant" and not m.tool_calls for m in self._messages)

    @property
    def needs_response(self) -> bool:
        return bool(self._messages and self._messages[-1].role == "tool")

    def begin_tools(self, content: str, calls: list[dict[str, Any]], thinking: str = "") -> int:
        """Record a complete assistant response and ordered, unexecuted placeholders."""
        start = len(self._messages)
        messages = [Message("assistant", content, tool_calls=deepcopy(calls),
                            thinking=thinking or None)]
        for call in calls:
            messages.append(Message(
                "tool", json.dumps({"status": "not_started",
                                    "detail": "Command has not been approved or executed."}),
                tool_name=call["function"]["name"], tool_call_id=call.get("id"),
            ))
        self._messages += messages
        return start + 1

    def set_tool_result(self, index: int, result: dict[str, Any]) -> None:
        previous = self._messages[index]
        if previous.role != "tool":
            raise ValueError("Expected a tool result")
        # Tool results are stored as JSON text; only replace unfinished placeholders.
        if json.loads(previous.content)["status"] not in ("not_started", "pending"):
            raise ValueError("Cannot replace a completed tool result")
        self._messages[index] = Message(
            "tool", json.dumps(result, ensure_ascii=False, allow_nan=False),
            tool_name=previous.tool_name, tool_call_id=previous.tool_call_id,
        )

    def add_message(self, role: Literal["assistant", "user", "tool"], content: str,
                    **metadata: Any) -> None:
        self._messages.append(Message(role, content=content, **deepcopy(metadata)))

    def rollback(self) -> bool:
        """Discard only an unfulfilled user prompt, never recorded command activity."""
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()
            return True
        return False

    def clear(self) -> None:
        self._messages.clear()
        self._new_identity()

    def get_preview(self, max_chars: int = 60) -> list[tuple[int, str, str]]:
        """Return (index, role, truncated_content) entries for history inspection."""
        previews: list[tuple[int, str, str]] = []
        for idx, msg in enumerate(self._messages, 1):
            clean = msg.content.replace("\n", " ").strip()
            snippet = clean[:max_chars] + "..." if len(clean) > max_chars else clean
            previews.append((idx, msg.role, snippet))
        return previews
