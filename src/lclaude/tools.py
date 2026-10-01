"""Read-only project tools. Full results and bounded excerpts are kept separately."""

import fnmatch
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CHARS = 4000
MAX_CHARS = 12000
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
ArtifactReader = Callable[[str, int, int], tuple[str, int]]


class ToolError(ValueError):
    """Invalid arguments, inaccessible files, or unsafe paths."""


@dataclass(frozen=True)
class ToolOutput:
    content: str
    max_chars: int = DEFAULT_CHARS
    max_lines: int | None = None
    artifact_id: str | None = None
    start: int = 0
    total_chars: int | None = None

    @property
    def excerpt(self) -> str:
        text = self.content
        if self.max_lines is not None:
            text = "".join(text.splitlines(keepends=True)[:self.max_lines])
        return text[:self.max_chars]

    @property
    def truncated(self) -> bool:
        total = len(self.content) if self.total_chars is None else self.total_chars - self.start
        return len(self.excerpt) < total


def _inside(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ToolError(f"Path is outside the project: {name}")
    return target


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.startswith("**/") and _matches(path, pattern[3:])
    )


def _files(root: Path, base: Path) -> Iterator[Path]:
    seen: set[Path] = set()

    def fail(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(base, followlinks=False, onerror=fail):
        resolved = Path(directory).resolve()
        if resolved in seen or not resolved.is_relative_to(root):
            dirs[:] = []
            continue
        seen.add(resolved)
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS
                         and (resolved / d).resolve().is_relative_to(root))
        for name in sorted(files):
            path = Path(directory) / name
            target = path.resolve()
            if target.is_relative_to(root) and target.is_file():
                yield path


def list_files(root: Path, path: str = ".", glob: str = "**/*",
               limit: int = 200, max_chars: int = DEFAULT_CHARS) -> ToolOutput:
    root = root.resolve()
    base = _inside(root, path)
    if not base.is_dir():
        raise ToolError(f"Directory not found: {path}")
    names = (p.relative_to(root).as_posix() for p in _files(root, base))
    content = "".join(name + "\n" for name in names if _matches(name, glob))
    return ToolOutput(content, max_chars, limit)


def read_file(root: Path, path: str, start_line: int = 1, end_line: int | None = None,
              max_chars: int = DEFAULT_CHARS) -> ToolOutput:
    target = _inside(root, path)
    if not target.is_file():
        raise ToolError(f"File not found: {path}")
    if start_line < 1 or (end_line is not None and end_line < start_line):
        raise ToolError("Invalid line range: use positive, inclusive line numbers")
    text = target.read_text(encoding="utf-8")
    if "\x00" in text:
        raise ToolError(f"File is not text: {path}")
    lines = text.splitlines(keepends=True)
    if start_line > max(1, len(lines)) or (end_line is not None and end_line > len(lines)):
        raise ToolError(f"Line range exceeds the file's {len(lines)} lines")
    return ToolOutput("".join(lines[start_line - 1:end_line]), max_chars)


def search_text(root: Path, query: str, glob: str = "**/*", limit: int = 100,
                max_chars: int = DEFAULT_CHARS) -> ToolOutput:
    root = root.resolve()
    matches: list[str] = []
    for path in _files(root, root):
        name = path.relative_to(root).as_posix()
        if not _matches(name, glob):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "\x00" in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if query.casefold() in line.casefold():
                matches.append(f"{name}:{number}:{line}\n")
    return ToolOutput("".join(matches), max_chars, limit)


def read_tool_output(reader: ArtifactReader, artifact_id: str, start: int = 0,
                     max_chars: int = DEFAULT_CHARS) -> ToolOutput:
    content, total = reader(artifact_id, start, max_chars)
    return ToolOutput(content, max_chars, artifact_id=artifact_id, start=start,
                      total_chars=total)


def _schema(name: str, description: str, properties: dict[str, Any],
            required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


_TEXT = {"type": "string", "minLength": 1}
_CHARS = {"type": "integer", "minimum": 1, "maximum": MAX_CHARS}
_LINE = {"type": "integer", "minimum": 1}
TOOL_SCHEMAS = [
    _schema("list_files", "List project files; large results are saved as artifacts.",
            {"path": _TEXT, "glob": _TEXT,
             "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
             "max_chars": _CHARS}, []),
    _schema("read_file", "Read UTF-8 text with optional inclusive, 1-based line ranges.",
            {"path": _TEXT, "start_line": _LINE, "end_line": _LINE, "max_chars": _CHARS},
            ["path"]),
    _schema("search_text", "Case-insensitive literal search of project UTF-8 text.",
            {"query": _TEXT, "glob": _TEXT,
             "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
             "max_chars": _CHARS}, ["query"]),
    _schema("read_tool_output", "Retrieve more saved output using its artifact ID and "
            "zero-based character offset. Use next_start from a truncated result.",
            {"artifact_id": _TEXT, "start": {"type": "integer", "minimum": 0},
             "max_chars": _CHARS}, ["artifact_id"]),
]


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    schema = next((s["function"]["parameters"] for s in TOOL_SCHEMAS
                   if s["function"]["name"] == name), None)
    if schema is None:
        raise ToolError(f"Unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise ToolError("Tool arguments must be a JSON object")
    if not all(isinstance(key, str) for key in arguments):
        raise ToolError("Tool argument names must be strings")
    missing = set(schema["required"]) - arguments.keys()
    unknown = arguments.keys() - schema["properties"].keys()
    if missing or unknown:
        raise ToolError(f"Invalid arguments; missing: {sorted(missing)}, "
                        f"unknown: {sorted(unknown)}")
    for key, value in arguments.items():
        rule = schema["properties"][key]
        if rule["type"] == "string":
            if not isinstance(value, str) or not value or "\x00" in value:
                raise ToolError(f"{key} must be a nonempty string without NUL characters")
        elif type(value) is not int or value < rule["minimum"] or value > rule.get(
            "maximum", 2**63 - 1
        ):
            raise ToolError(f"{key} must be an integer in the allowed range")
    return arguments


def dispatch(root: Path, name: str, arguments: Any,
             artifact_reader: ArtifactReader | None = None) -> ToolOutput:
    arguments = validate_arguments(name, arguments)
    try:
        if name == "list_files":
            return list_files(root, **arguments)
        if name == "read_file":
            return read_file(root, **arguments)
        if name == "search_text":
            return search_text(root, **arguments)
        if artifact_reader is None:
            raise ToolError("Artifact storage is unavailable")
        return read_tool_output(artifact_reader, **arguments)
    except (OSError, UnicodeError, RuntimeError) as exc:
        raise ToolError(f"{name} failed: {exc}") from exc
