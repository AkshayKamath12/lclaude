# Command tool loop

lclaude uses the [Ollama Python SDK's structured tool calling](https://docs.ollama.com/capabilities/tool-calling).
The flow is:

```text
stream model response → validate call → ask for approval
  → save pending execution → run command → save result → next model request
```

The CLI coordinates the loop. `engine.py` owns SDK transport and streaming;
`tools.py` validates requests and runs subprocesses; `session.py` owns ordered
messages; `persistence.py` atomically saves them; `ui.py` owns approval and rendering.
There is no separate tool ledger or generic dispatch framework.

## One tool: run_command

| Argument | Meaning |
| --- | --- |
| `command` | Required nonempty shell script. |
| `shell` | Required: `powershell` on Windows; `sh` on Linux/macOS. |
| `cwd` | Existing directory, relative to the launch directory or absolute. Default: `.`. |
| `timeout_seconds` | Integer from 1 to 300. Default: 60. Separate from the Ollama socket timeout. |

The tool definition tells the model to emit a structured call in the same response
when asked to run a command. A code block or a statement such as “I'll run this”
does not execute anything; only a matching saved tool result is evidence that a
command ran. Each call starts a new shell process. A `cd` inside one call does not
carry over; pass the desired `cwd` on every call.

On Windows, commands run through Windows PowerShell
(`System32/WindowsPowerShell/v1.0/powershell.exe`) with no profile and noninteractive
input. On Linux/macOS, commands run through `/bin/sh -c`. Bash and PowerShell syntax
are not interchangeable: unsupported shell requests are rejected, never translated.
PowerShell's console output encoding is set to UTF-8; captured bytes are decoded
as UTF-8 with replacement for invalid sequences.

Before every execution the terminal shows the **entire command**, shell, resolved
directory, and timeout. Control characters are escaped visibly. Only an explicit
`y` or `yes` approves; Enter, EOF, and other answers reject. Ctrl+C cancels.
Redirected input cannot approve commands. Approval applies to that one call only.

Commands run with the user's permissions. **The working directory is not a security
sandbox.** A command can modify files, access the network, or start other processes.
There is no automatic approval or trusted mode in this change.

## Streaming, ordering, and limits

The assistant's text streams normally. Thinking text and structured tool calls
are accumulated alongside it. Nothing executes until the SDK stream finishes
with a completion marker. Partial assistant messages are never saved as complete.

Calls from one response execute sequentially, each with its own approval.
The complete assistant call list is followed by matching `role: "tool"` messages
in the original order. Thinking fields are retained for subsequent requests.
The SDK's message shape is used directly; no synthetic call IDs are required.
Historical IDs remain in saved chats when importing older versions.

The loop requests another model response after saving every result, including
rejections and execution errors. Cancellation stops the turn. The default
`--max-tool-iterations 10` counts model responses per user submission, including
the final response. At the limit, any calls in that last response are processed
and saved, then the loop stops. It does not automatically continue on resume.

Stdout and stderr each retain at most **6,000 bytes**. Both pipes are drained
concurrently so large output cannot fill a pipe and deadlock the command.
Excess output is discarded. The result includes a truncation flag and explicit
notice that omitted output was not saved. The **same bounded JSON result** is
saved and sent to the model. There are no artifacts, retrieval tools, or rules
withholding final answers until output has been fully read.

Results distinguish success, nonzero exit, execution/capture error, timeout,
rejection, and cancellation. Completed subprocess results include exit code,
stdout, stderr, and truncation state. Ctrl+C and timeout attempt to stop the
process tree on Windows (`taskkill /T /F`) or process group on POSIX. Cleanup
failures are reported. Deliberately detached processes can escape cleanup;
commands are not contained by the application. Partial effects are never undone.

## Persistence and resume

Before approval, the complete call bundle is saved with `not_started` placeholders.
After approval, the current call becomes `pending` and is durably saved **before**
the subprocess starts. That record includes the resolved directory and shell.
After execution, its placeholder is replaced by the bounded result and saved
**before** another command or model request. A failed checkpoint stops the loop.

- `not_started`: execution was not started by the application.
- `pending`: execution may have started or finished, but no completed outcome was saved.
- Completed status: the saved result describes what the application observed.

A crash can happen after an external effect but before its result is saved.
Consequently, pending commands have an **unknown outcome**; they are never
automatically rerun or described as successfully rolled back. A later inference
failure cannot remove recorded command activity. Resuming displays saved activity,
explains these statuses, and waits for the user's next message.

An interrupted initial chat response still discards its unfulfilled user prompt.
Atomic snapshots use a temporary file, flush/fsync, then replacement. Command
execution and file replacement cannot form one transaction, hence the pending state.

## Schema version 3

V3 stores messages close to the SDK format. Result status lives inside the JSON
`content` of its tool message; no parallel ledger or artifact manifest is written.

```json
{
  "schema_version": 3,
  "session_id": "a65e3b8b-f25d-4c04-9235-8b2d5d2f8850",
  "project": {"path": "/work/demo"},
  "created_at": "2026-10-02T18:00:00+00:00",
  "updated_at": "2026-10-02T18:00:05+00:00",
  "model": "qwen3",
  "messages": [
    {"role": "user", "content": "Print hello."},
    {
      "role": "assistant",
      "content": "I will print hello.",
      "thinking": "A short command will do.",
      "tool_calls": [
        {"function": {"name": "run_command", "arguments": {"command": "printf hello", "shell": "sh"}}}
      ]
    },
    {
      "role": "tool",
      "tool_name": "run_command",
      "content": "{\"status\":\"success\",\"detail\":\"Command completed.\",\"exit_code\":0,\"stdout\":\"hello\",\"stderr\":\"\",\"truncated\":false}"
    },
    {"role": "assistant", "content": "The command printed hello."}
  ]
}
```

Version-1 plain chats and version-2 read-only tool chats remain readable. V2 call
and result fields are converted on load; old receipts remain historical text.
Interrupted/incomplete legacy results become pending with an unknown-outcome
explanation. Existing artifact files are left untouched but are no longer
accessed; their manifest is omitted on the next save, which writes v3.
Simply opening a chat does not rewrite it.

## Context budget

Every inference iteration counts instructions, message content, thinking,
tool-call arguments, results, tool definitions, and message overhead. The existing
conservative estimate plus reserved reply tokens must fit the context allocation.
The transcript is never compacted, summarized, or stripped of call/result pairs.

An oversized request stops with a clear error; recorded command activity remains
saved. Increase `--num-ctx`, reduce `--max-response-tokens`, or start a fresh chat.
If the runtime allocation is unknown, lclaude attempts to load the model and
query it. If it remains unknown, set `--num-ctx` explicitly. Counts are estimates,
not a model-specific tokenizer or a guarantee against server-side truncation.
