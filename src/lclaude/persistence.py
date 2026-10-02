"""Versioned chat serialization and atomic project-local storage."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from lclaude.session import Session


class SessionStorageError(Exception):
    """A saved chat could not be validated, read, or written."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SessionStorageError(message)


def _session_id(value: object) -> str:
    _require(isinstance(value, str), "session_id must be a UUID string")
    try:
        result = str(UUID(str(value)))
    except ValueError as exc:
        raise SessionStorageError("session_id must be a UUID string") from exc
    _require(result == value, "session_id must be a canonical UUID")
    return result


def _timestamp(value: object, field: str) -> datetime:
    _require(isinstance(value, str), f"{field} must be an ISO 8601 timestamp")
    try:
        result = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise SessionStorageError(f"{field} must be an ISO 8601 timestamp") from exc
    _require(result.tzinfo is not None, f"{field} must include a timezone")
    return result


def deserialize(data: Any, project: Path) -> tuple[Session, str]:
    """Read v1/v2 chats and the current SDK-shaped v3 transcript."""
    _require(isinstance(data, dict), "session must be a JSON object")
    version = data.get("schema_version")
    _require(type(version) is int and version in (1, 2, 3),
             f"Unsupported schema_version: {version!r}; expected 1, 2 or 3")
    fields = {"schema_version", "session_id", "project", "created_at",
              "updated_at", "model", "messages"}
    if version == 2:
        fields.add("artifacts")
    _require(set(data) - {"system_prompt"} == fields, "Invalid schema fields")
    session_id = _session_id(data["session_id"])
    _require(data["project"] == {"path": str(project)}, "Session belongs to a different project")
    created = _timestamp(data["created_at"], "created_at")
    updated = _timestamp(data["updated_at"], "updated_at")
    _require(updated >= created, "updated_at precedes created_at")
    model = data["model"]
    _require(isinstance(model, str) and bool(model.strip()), "model must be a nonempty string")
    messages = data["messages"]
    _require(isinstance(messages, list), "messages must be a list")
    if version == 1:
        _require(len(messages) % 2 == 0, "messages must contain complete user/assistant turns")
    if version == 2:
        _require(isinstance(data["artifacts"], list), "artifacts must be a list")
    session = Session()
    session.session_id = session_id
    session.created_at = data["created_at"]
    session.updated_at = data["updated_at"]
    pending: list[dict[str, Any]] = []
    previous_role: str | None = None
    for index, original in enumerate(messages):
        _require(isinstance(original, dict), f"Invalid message at index {index}")
        message = dict(original)
        role = message.get("role")
        _require(role in ("user", "assistant", "tool"), f"Invalid role at index {index}")
        _require(isinstance(message.get("content"), str),
                 f"Message {index} must have string content")
        if version == 1:
            expected = "user" if index % 2 == 0 else "assistant"
            _require(role == expected and set(message) == {"role", "content"},
                     f"Message {index} must have role {expected!r}")
        if version == 2:
            # Import historical receipts as text. Artifact files are left untouched,
            # but are no longer read or written by the application.
            if role == "assistant" and "tool_calls" in message:
                calls = message["tool_calls"]
                _require(isinstance(calls, list) and bool(calls), "Invalid tool_calls")
                converted = []
                for call in calls:
                    _require(isinstance(call, dict)
                             and set(call) == {"id", "name", "arguments"},
                             "Invalid legacy assistant tool call")
                    converted.append({"id": call["id"], "function": {
                        "name": call["name"], "arguments": call["arguments"]}})
                message["tool_calls"] = converted
            if role == "tool":
                _require({"role", "content", "tool_call_id", "name", "status"} <= set(message),
                         "Invalid legacy tool result")
                status = message.pop("status")
                _require(status in ("success", "error", "incomplete", "interrupted"),
                         "Invalid legacy result status")
                message["tool_name"] = message.pop("name")
                message.pop("artifact_id", None)
                try:
                    result = json.loads(message["content"])
                except ValueError:
                    result = {"detail": message["content"]}
                if not isinstance(result, dict):
                    result = {"detail": message["content"]}
                result["status"] = "pending" if status in ("incomplete", "interrupted") else status
                if result["status"] == "pending":
                    result["detail"] = "Legacy interrupted call; execution outcome is unknown."
                message["content"] = json.dumps(result, ensure_ascii=False)
        if role == "user":
            _require(not pending and previous_role in (None, "assistant", "tool")
                     and set(message) == {"role", "content"}, "Invalid user role or fields")
        elif role == "assistant":
            _require(not pending and previous_role in ("user", "tool"),
                     "Invalid assistant role or unresolved tool calls")
            _require(set(message) <= {"role", "content", "thinking", "tool_calls"},
                     "Invalid assistant fields")
            if "thinking" in message:
                _require(isinstance(message["thinking"], str), "thinking must be a string")
            if "tool_calls" in message:
                calls = message["tool_calls"]
                _require(isinstance(calls, list) and bool(calls), "Invalid tool_calls")
                ids: set[str] = set()
                for call in calls:
                    _require(isinstance(call, dict)
                             and set(call) <= {"function", "id", "type"}, "Invalid tool call")
                    function = call.get("function")
                    _require(isinstance(function, dict)
                             and set(function) <= {"name", "arguments", "index"}
                             and isinstance(function.get("name"), str), "Invalid tool function")
                    if "id" in call:
                        call_id = call["id"]
                        _require(isinstance(call_id, str) and bool(call_id)
                                 and call_id not in ids, "Invalid or duplicate call ID")
                        ids.add(call_id)
                    try:
                        json.dumps(call, allow_nan=False)
                    except (TypeError, ValueError) as exc:
                        raise SessionStorageError("Tool calls must be JSON values") from exc
                pending = list(calls)
        else:
            _require({"role", "content", "tool_name"} <= set(message)
                     <= {"role", "content", "tool_name", "tool_call_id"},
                     "Invalid tool result fields")
            _require(bool(pending), "Orphaned tool result")
            call = pending.pop(0)
            _require(message["tool_name"] == call["function"]["name"]
                     and message.get("tool_call_id") == call.get("id"),
                     "Tool results must match calls in their original order")
            try:
                result = json.loads(message["content"])
            except ValueError as exc:
                raise SessionStorageError("Invalid tool result JSON") from exc
            _require(isinstance(result, dict) and result.get("status") in (
                "not_started", "pending", "success", "error", "rejected",
                "nonzero_exit", "timeout", "cancelled",
            ), "Invalid result status")
        session.add_message(
            cast(Literal["user", "assistant", "tool"], role), message["content"],
            **{k: v for k, v in message.items() if k not in ("role", "content")},
        )
        previous_role = role
    _require(not pending and previous_role != "user",
             "Session must contain complete responses or complete tool exchanges")
    return session, str(model)


def serialize(session: Session, model: str, project: Path, updated_at: str) -> dict[str, Any]:
    """Serialize full history independently of inference context and presentation."""
    data = {
        "schema_version": 3,
        "session_id": session.session_id,
        "project": {"path": str(project)},
        "created_at": session.created_at,
        "updated_at": updated_at,
        "model": model,
        "messages": session.conversation,
    }
    deserialize(data, project)
    return data


def _atomic_write(destination: Path, data: dict[str, Any]) -> None:
    """Replace the saved file only after the complete snapshot is safely written."""
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A leftover temp is never listed or resumed as a conversation.
                pass


class ChatStore:
    """One UUID-named JSON file per session, grouped by the resolved launch path."""

    def __init__(self, project: Path, root: Path | None = None) -> None:
        self.project = project.resolve()
        root = root if root is not None else Path.home() / ".local_claude" / "chats"
        project_key = hashlib.sha256(str(self.project).encode("utf-8")).hexdigest()
        self.directory = root / project_key

    def load(self, session_id: str) -> tuple[Session, str]:
        path = self.directory / f"{_session_id(session_id)}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session, model = deserialize(data, self.project)
            _require(session.session_id == session_id, "session_id does not match filename")
            return session, model
        except (OSError, ValueError, SessionStorageError) as exc:
            raise SessionStorageError(f"Cannot load '{path}': {exc}") from exc

    def list_sessions(self) -> tuple[list[tuple[Session, str]], list[str]]:
        """List valid sessions newest first and report invalid files individually."""
        sessions: list[tuple[Session, str]] = []
        errors: list[str] = []
        try:
            paths = [path for path in self.directory.iterdir() if path.suffix == ".json"]
        except FileNotFoundError:
            return sessions, errors
        except OSError as exc:
            raise SessionStorageError(f"Cannot list '{self.directory}': {exc}") from exc
        for path in paths:
            try:
                chat, model = self.load(path.stem)
                if not chat.is_empty:
                    sessions.append((chat, model))
            except SessionStorageError as exc:
                errors.append(f"{path.name}: {exc}")
        sessions.sort(
            key=lambda item: (
                datetime.fromisoformat(item[0].updated_at), item[0].session_id
            ),
            reverse=True,
        )
        return sessions, errors

    def latest_session(self) -> tuple[tuple[Session, str] | None, list[str]]:
        """Find the most recently updated nonempty chat in this project."""
        sessions, errors = self.list_sessions()
        return (sessions[0] if sessions else None), errors

    def save(self, session: Session, model: str) -> None:
        """Flush a complete snapshot before replacing its destination atomically."""
        if session.is_empty:
            return
        updated_at = datetime.now(timezone.utc).isoformat()
        data = serialize(session, model, self.project, updated_at)
        destination = self.directory / f"{session.session_id}.json"
        try:
            if destination.exists():
                # Validate before replacing, and preserve recency for unchanged chats.
                previous, _ = self.load(session.session_id)
                previous_data = json.loads(destination.read_text(encoding="utf-8"))
                candidate = dict(data, updated_at=previous.updated_at)
                if candidate == previous_data:
                    session.updated_at = previous.updated_at
                    return
            _atomic_write(destination, data)
            session.updated_at = updated_at
        except (OSError, ValueError, SessionStorageError) as exc:
            raise SessionStorageError(f"Cannot save '{destination}': {exc}") from exc

    def save_model(self, session_id: str, model: str) -> None:
        """Update only saved model metadata, preserving history and timestamps."""
        session_id = _session_id(session_id)
        destination = self.directory / f"{session_id}.json"
        try:
            _, previous_model = self.load(session_id)
            if previous_model == model:
                return
            data = json.loads(destination.read_text(encoding="utf-8"))
            data["model"] = model
            deserialize(data, self.project)
            _atomic_write(destination, data)
        except (OSError, ValueError, SessionStorageError) as exc:
            raise SessionStorageError(f"Cannot save model for '{destination}': {exc}") from exc
