"""Session management for lclaude"""

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
    tool_call_id: str | None = None
    name: str | None = None
    status: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        for key in ("tool_calls", "tool_call_id", "name", "status", "artifact_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return deepcopy(result)


class Session:
    """Manages conversational state, turn tracking, and history rollbacks."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt
        self._new_identity()
        self._messages: list[Message] = []
        self.artifacts: list[dict[str, Any]] = []

    def _new_identity(self) -> None:
        """Refresh identity and timestamps so the next save records a new chat."""
        self.session_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

    @property
    def conversation(self) -> list[dict[str, Any]]:
        """Full conversation, excluding the separately stored system prompt."""
        return [message.to_dict() for message in self._messages]

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Returns messages formatted for Ollama API consumption."""
        payload: list[dict[str, Any]] = []
        if self.system_prompt:
            payload.append({"role": "system", "content": self.system_prompt})
        for message in self._messages:
            item = message.to_dict()
            if item.get("tool_calls"):
                item["tool_calls"] = [{"id": call["id"], "type": "function",
                                       "function": {"name": call["name"],
                                                    "arguments": call["arguments"]}}
                                      for call in item["tool_calls"]]
            if item["role"] == "tool":
                item = {"role": "tool", "content": item["content"],
                        "tool_name": item["name"], "tool_call_id": item["tool_call_id"]}
            if item["role"] == "user" and payload and payload[-1]["role"] == "tool":
                payload.append({"role": "system", "content":
                    "The previous turn stopped after tool work. Its saved results are above. "
                    "Do not automatically repeat completed calls; follow the user's next request."})
            payload.append(item)
        return payload

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def turn_count(self) -> int:
        """Returns the number of completed user-assistant round trips."""
        return sum(m.role == "assistant" and not m.tool_calls for m in self._messages)

    @property
    def needs_response(self) -> bool:
        """Whether the saved turn needs recovery or an assistant continuation."""
        if not self._messages:
            return False
        last = self._messages[-1]
        return bool(
            last.role == "tool"
            or (last.role == "assistant" and last.tool_calls)
        )

    def begin_tools(self, content: str, calls: list[dict[str, Any]]) -> int:
        """Publish one complete exchange, initially with explicit unexecuted results.

        A checkpoint made before execution is valid even after process termination.
        Each completed result replaces its placeholder; calls are never replayed on load.
        """
        start = len(self._messages)
        messages = [Message("assistant", content, tool_calls=deepcopy(calls))]
        for call in calls:
            messages.append(Message(
                "tool", json.dumps({"error": "No completed result was saved. "
                                    "Work may have stopped before or during this call."}),
                tool_call_id=call["id"], name=call["name"], status="interrupted",
            ))
        self._messages = self._messages + messages
        return start + 1

    def complete_tool(self, index: int, content: str, status: str,
                      artifact_id: str | None = None) -> None:
        previous = self._messages[index]
        if previous.role != "tool" or previous.status != "interrupted":
            raise ValueError("Tool result is already complete or is not a pending result")
        self._messages[index] = Message(
            "tool", content, tool_call_id=previous.tool_call_id, name=previous.name,
            status=status, artifact_id=artifact_id,
        )

    def add_message(self, role: Literal["assistant", "user", "tool"], content: str,
                    **metadata: Any) -> None:
        self._messages.append(Message(role, content=content, **deepcopy(metadata)))

    def rollback(self) -> bool:
        "on generation interrupt, removed last message if from user prompt"
        if self._messages and self._messages[-1].role == "user":
            self._messages.pop()
            return True
        return False

    def clear(self) -> None:
        self._messages.clear()
        self.artifacts.clear()
        self._new_identity()

    def get_preview(self, max_chars: int = 60) -> list[tuple[int, str, str]]:
        """Returns (index, role, truncated_content) for history inspection."""
        previews: list[tuple[int, str, str]] = []
        for idx, msg in enumerate(self._messages, 1):
            clean_content = msg.content.replace("\n", " ").strip()
            snippet = (
                clean_content[:max_chars] + "..."
                if len(clean_content) > max_chars
                else clean_content
            )
            previews.append((idx, msg.role, snippet))
        return previews
