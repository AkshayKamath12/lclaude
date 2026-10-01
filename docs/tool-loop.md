# Read-only tool loop

lclaude sends explicit JSON tool definitions using [Ollama structured tool calling](https://docs.ollama.com/capabilities/tool-calling).
A model response may request several tools. lclaude validates each request, executes
calls in their original order, then sends the results in the next inference request.
Ordinary responses still stream. A stream must end with Ollama's completion marker
before any assistant message or tool calls are accepted.

The application loop coordinates these steps. Tools perform project reads; the
inference engine handles Ollama transport; session state owns messages; the store
owns artifacts and atomic snapshots; the terminal UI renders activity.

## Tools and bounds

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `list_files` | `path="."`, `glob="**/*"`, `limit=200`, `max_chars=4000` | Discover project files. The excerpt contains at most `limit` paths. |
| `read_file` | `path`, `start_line=1`, optional `end_line`, `max_chars=4000` | Read UTF-8 text. Line numbers are inclusive and start at 1. |
| `search_text` | `query`, `glob="**/*"`, `limit=100`, `max_chars=4000` | Case-insensitive literal search. The excerpt contains at most `limit` matches with file and line locations. |
| `read_tool_output` | `artifact_id`, `start=0`, `max_chars=4000` | Retrieve saved text using a zero-based character offset. |

`max_chars` must be between 1 and 12,000. List limits range from 1 to 2,000;
search limits range from 1 to 1,000. Unknown arguments, wrong types, missing
files, and invalid ranges produce explicit error results for the model.

Relative paths use the resolved launch directory. Absolute paths must remain
inside it. Traversal and links outside the project are rejected or excluded from
discovery. Recursive discovery excludes `.git`, `.venv`, `venv`, `node_modules`,
and `__pycache__`. Glob matching is case-sensitive and uses project-relative
paths; `**/*` also includes root-level files. Git ignore rules are not applied.
Search skips non-UTF-8 and NUL-containing files; explicitly reading them returns
an error. Filesystem permission failures are reported.

Each executed result, including errors, has a complete UTF-8 artifact. Excerpt
limits do not discard the rest of a listing, search result, or selected file
range. Retrieval reuses the original artifact ID. A result includes its excerpt,
status, truncation flag, artifact ID, character offsets, and retrieval instructions.
The model can pass `next_start` as `start` to retrieve the next portion.

The iteration limit counts model responses per submitted user message, including
the final answer. The default is 10. Reaching it stops after saving completed
tool work; the next user message starts a new allowance.

## Schema version 2

The session JSON retains the original ordered transcript. Assistant messages
keep the complete call list from each response. Each call has a matching result
in the same order. Server-provided IDs are preserved; absent IDs receive stable
internal UUIDs. Raw JSON chat streaming avoids SDK conversion that would drop
optional IDs or malformed arguments before validation.

Version-1 sessions are loaded without rewriting them. Their next save uses v2.
Artifacts live under `<project-store>/<session-id>/artifacts/<artifact-id>.txt`
and are accessible only through that session's manifest.

This complete example references an artifact containing `one\ntwo\nthree\n`:

```json
{
  "schema_version": 2,
  "session_id": "a65e3b8b-f25d-4c04-9235-8b2d5d2f8850",
  "project": {"path": "C:\\work\\demo"},
  "created_at": "2026-10-01T18:00:00+00:00",
  "updated_at": "2026-10-01T18:00:05+00:00",
  "model": "qwen2.5:7b-instruct",
  "messages": [
    {"role": "user", "content": "Read the notes."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {"id": "call_01", "name": "read_file", "arguments": {"path": "notes.txt", "max_chars": 4}}
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_01",
      "name": "read_file",
      "status": "success",
      "artifact_id": "art_0123456789abcdef0123456789abcdef",
      "content": "{\"status\":\"success\",\"excerpt\":\"one\\n\",\"truncated\":true,\"artifact_id\":\"art_0123456789abcdef0123456789abcdef\",\"start\":0,\"next_start\":4,\"total_chars\":14,\"retrieval\":\"Use read_tool_output with artifact_id and start=next_start.\"}"
    },
    {"role": "assistant", "content": "The first line is one; further lines are available in the saved output."}
  ],
  "artifacts": [
    {
      "id": "art_0123456789abcdef0123456789abcdef",
      "path": "artifacts/art_0123456789abcdef0123456789abcdef.txt",
      "media_type": "text/plain",
      "size_bytes": 14,
      "size_chars": 14
    }
  ]
}
```

## Interruption and resume

Before executing a batch, lclaude atomically saves the complete assistant call
list and one `interrupted` result placeholder per call. This is a recovery
checkpoint, never an inference request with missing results. A placeholder says
that no completed result was saved: execution may not have started, or may have
been interrupted.

After each call, lclaude flushes and fsyncs its full artifact, replaces the result
placeholder, and atomically saves the session before starting the next call.
Completed result statuses are `success` or `error`. A saving failure stops the
loop and is shown to the user; the last durable checkpoint remains readable.

Ctrl+C or a connection failure during subsequent inference discards the partial
assistant response. Already saved tool results remain paired with their calls.
An interrupted initial inference still rolls back its unfulfilled user message.
Unexpected process termination leaves the last checkpoint intact. A crash before
a result checkpoint may leave that call marked interrupted and an unreferenced
artifact on disk.

Startup and `/chat` replay saved tool statuses and explain unfinished work.
They neither generate a response nor repeat calls. The next user message supplies
direction, with a context notice that completed calls should not be repeated
automatically. `/clear` creates a new identity and manifest while retaining
earlier saved sessions and artifacts.

## Context and practical limits

The token estimate includes tool definitions, assistant arguments and identifiers,
results, retrieval excerpts, and continuation notices. Prompt assembly works on
copies. For tool conversations with a known context limit, it reduces retrievable
excerpts and preserves their artifact receipts until the estimated prompt plus
reserved reply tokens fits. It never removes messages or call/result pairs.
If even minimal receipts cannot fit, inference stops with an actionable error.
There is no summarization. The saved transcript remains unchanged by assembly.

Counts are estimates, not an exact model tokenizer. When the runtime context
size is unknown, excerpt limits still apply but fit cannot be guaranteed.
Full reads and searches can consume time, RAM, and disk proportional to their
complete output. Artifacts have no automatic retention or quota policy.
Path containment is checked using resolved paths; it is not an OS security
sandbox against concurrent filesystem changes.
