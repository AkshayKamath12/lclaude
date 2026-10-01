"""Versioned chat serialization and atomic project-local storage."""

import hashlib
import json
import os
import re
import secrets
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
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
    """Validate v1 and v2 before constructing any active conversation state."""
    _require(isinstance(data, dict), "session must be a JSON object")
    version = data.get("schema_version")
    _require(type(version) is int and version in (1, 2),
             f"Unsupported schema_version: {version!r}; expected 1 or 2")
    # Accept the earlier draft field when reading, but never restore or write it.
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
    session = Session()
    session.session_id = session_id
    session.created_at = data["created_at"]
    session.updated_at = data["updated_at"]
    artifacts = data.get("artifacts", [])
    _require(isinstance(artifacts, list), "artifacts must be a list")
    artifact_ids: set[str] = set()
    for entry in artifacts:
        _require(isinstance(entry, dict), "Invalid artifact")
        _require(set(entry) == {"id", "path", "media_type", "size_bytes", "size_chars"},
                 "Invalid artifact fields")
        artifact_id = entry["id"]
        _require(isinstance(artifact_id, str) and re.fullmatch(r"art_[0-9a-f]{32}", artifact_id)
                 is not None and artifact_id not in artifact_ids, "Invalid artifact ID")
        _require(entry["path"] == f"artifacts/{artifact_id}.txt"
                 and entry["media_type"] == "text/plain", "Invalid artifact path or media type")
        _require(all(type(entry[k]) is int and entry[k] >= 0
                     for k in ("size_bytes", "size_chars")), "Invalid artifact size")
        artifact_ids.add(artifact_id)
    session.artifacts = deepcopy(artifacts)
    pending: list[dict[str, Any]] = []
    previous_role: str | None = None
    for index, message in enumerate(messages):
        _require(isinstance(message, dict), f"Invalid message at index {index}")
        role = message.get("role")
        _require(role in ("user", "assistant", "tool"), f"Invalid role at index {index}")
        _require(isinstance(message.get("content"), str),
                 f"Message {index} must have string content")
        if version == 1:
            expected = "user" if index % 2 == 0 else "assistant"
            _require(role == expected and set(message) == {"role", "content"},
                     f"Message {index} must have role {expected!r}")
        elif role == "user":
            _require(not pending and previous_role in (None, "assistant", "tool")
                     and set(message) == {"role", "content"}, "Invalid user role or fields")
        elif role == "assistant":
            _require(not pending and previous_role in ("user", "tool"),
                     "Invalid assistant role or unresolved tool calls")
            _require(set(message) in ({"role", "content"}, {"role", "content", "tool_calls"}),
                     "Invalid assistant fields")
            if "tool_calls" in message:
                calls = message["tool_calls"]
                _require(isinstance(calls, list) and bool(calls), "Invalid tool_calls")
                ids: set[str] = set()
                for call in calls:
                    _require(isinstance(call, dict) and set(call) == {"id", "name", "arguments"},
                             "Invalid assistant tool call")
                    _require(isinstance(call["id"], str) and bool(call["id"])
                             and call["id"] not in ids, "Invalid or duplicate call ID")
                    _require(isinstance(call["name"], str), "Invalid tool name")
                    try:
                        json.dumps(call["arguments"], allow_nan=False)
                    except (TypeError, ValueError) as exc:
                        raise SessionStorageError("Arguments must be JSON values") from exc
                    ids.add(call["id"])
                pending = list(calls)
        else:
            required = {"role", "content", "tool_call_id", "name", "status"}
            _require(required <= set(message) <= required | {"artifact_id"},
                     "Invalid tool result fields")
            _require(bool(pending), "Orphaned tool result")
            call = pending.pop(0)
            _require(message["tool_call_id"] == call["id"] and message["name"] == call["name"],
                     "Tool results must match calls in their original order")
            _require(message["status"] in ("success", "error", "interrupted"),
                     "Invalid result status")
            if "artifact_id" in message:
                _require(isinstance(message["artifact_id"], str)
                         and message["artifact_id"] in artifact_ids, "Unknown artifact reference")
        session.add_message(role, message["content"], **{
            k: v for k, v in message.items() if k not in ("role", "content")
        })
        previous_role = role
    _require(not pending and previous_role != "user",
             "Session must contain complete responses or complete tool exchanges")
    return session, str(model)


def serialize(session: Session, model: str, project: Path, updated_at: str) -> dict[str, Any]:
    """Serialize full history independently of inference context and presentation."""
    data = {
        "schema_version": 2,
        "session_id": session.session_id,
        "project": {"path": str(project)},
        "created_at": session.created_at,
        "updated_at": updated_at,
        "model": model,
        "messages": session.conversation,
        "artifacts": deepcopy(session.artifacts),
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

    def _artifact_path(self, session: Session, artifact_id: str) -> Path:
        _require(isinstance(artifact_id, str)
                 and re.fullmatch(r"art_[0-9a-f]{32}", artifact_id) is not None,
                 "Invalid artifact ID")
        session_directory = self.directory.resolve() / _session_id(session.session_id)
        directory = session_directory / "artifacts"
        target = directory / f"{artifact_id}.txt"
        _require(target.resolve().is_relative_to(session_directory),
                 "Artifact path is outside its session")
        _require(target.resolve().is_relative_to(directory.resolve()),
                 "Artifact path is outside its session")
        return target

    def save_artifact(self, session: Session, content: str) -> str:
        """Durably write complete UTF-8 output before publishing its reference."""
        artifact_id = "art_" + secrets.token_hex(16)
        target = self._artifact_path(session, artifact_id)
        temporary: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=target.parent,
                suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            raise SessionStorageError(f"Cannot save tool output: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        session.artifacts.append({
            "id": artifact_id, "path": f"artifacts/{artifact_id}.txt",
            "media_type": "text/plain", "size_bytes": len(content.encode("utf-8")),
            "size_chars": len(content),
        })
        return artifact_id

    def read_artifact(self, session: Session, artifact_id: str, start: int = 0,
                      max_chars: int = 4000) -> tuple[str, int]:
        """Read a bounded character range, returning content and total character count."""
        _require(type(start) is int and start >= 0 and type(max_chars) is int
                 and 1 <= max_chars <= 12000, "Invalid artifact range")
        entry = next((a for a in session.artifacts if a["id"] == artifact_id), None)
        _require(entry is not None, f"Artifact not found in this session: {artifact_id}")
        assert entry is not None
        _require(start <= entry["size_chars"], "Artifact range starts beyond the output")
        target = self._artifact_path(session, artifact_id)
        try:
            _require(target.stat().st_size == entry["size_bytes"],
                     "Artifact size differs from its recorded size")
            with target.open(encoding="utf-8", newline="") as stream:
                remaining = start
                while remaining:
                    discarded = stream.read(min(remaining, 8192))
                    if not discarded:
                        raise SessionStorageError("Artifact is shorter than its recorded size")
                    remaining -= len(discarded)
                content = stream.read(max_chars)
            return content, entry["size_chars"]
        except (OSError, UnicodeError) as exc:
            raise SessionStorageError(f"Cannot read artifact {artifact_id}: {exc}") from exc

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
            for entry in session.artifacts:
                path = self._artifact_path(session, entry["id"])
                _require(path.is_file() and path.stat().st_size == entry["size_bytes"],
                         f"Artifact missing or incomplete: {entry['id']}")
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
