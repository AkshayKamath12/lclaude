# lclaude

Chat with local AI models in your terminal through Ollama. Keep your prompts on
your machine, without a cloud AI subscription.

Early stage: streaming chat with conversation history during the current session.
Coding tools and saved sessions are planned.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
[Git](https://git-scm.com/downloads), and [Ollama](https://ollama.com/download).
Requires Python 3.10+ (uv can install it automatically).

With Ollama running:

```sh
ollama pull qwen2.5:7b-instruct
uv tool install git+https://github.com/AkshayKamath12/lclaude.git
lclaude
```

By default, lclaude uses `qwen2.5:7b-instruct` if installed, otherwise the first
installed model alphabetically. If no models are installed, startup shows a
download instruction. Use `lclaude --model <model-name>` to require a specific
downloaded model; an unavailable explicit selection produces an error.

During a session, `/model` opens a picker of models available on the configured
Ollama server. Use **Up/Down** and **Enter** to select; **Escape** or **Ctrl+C**
cancels. `/model <model-name>` switches directly. Selection preserves conversation
history and applies to the next prompt. If validation fails, the previous model
stays selected. Type `/model ` followed by part of a name to see matching models;
use **Up/Down** to navigate and **Tab** to complete or **Enter** to select.
The startup model list is cached for the process lifetime and also used for
selection validation. Restart lclaude to pick up newly installed models. Models
removed after startup may fail on the next inference request.
Models are not downloaded or preloaded by this command.
With redirected input or output, `/model` lists available models without reading
another input line; use `/model <model-name>` to select one.

## Project instructions

At startup, lclaude reads `AGENTS.md` from the directory where you launch it,
falling back to `agents.md` when the uppercase filename is absent. It sends the
file contents as the system prompt on every request. Parent and child directories
are not searched. A missing or blank file uses a small default assistant prompt;
an unreadable file or invalid UTF-8 stops startup with an error.

Instructions stay loaded through `/clear` and model changes. Restart lclaude
to start a new chat after editing the file. This also applies with redirected input. Files are loaded
in full; context estimates are advisory and never prevent generation.

## Writing prompts

- **Enter** submits the complete prompt. Blank or whitespace-only input stays in
  the editor without being submitted.
- **Alt+Enter** inserts a newline. If your terminal intercepts that shortcut,
  press **Escape, then Enter** in sequence instead. On macOS, Option may need to
  be configured as Alt/Meta in your terminal.
- **Up/Down** move between logical lines in a multiline prompt. At the first or
  last logical line they recall older/newer submissions. A long line that wraps
  on screen still counts as one logical line. Returning past the newest entry
  restores your draft.
- Multiline paste preserves indentation and line breaks when the terminal
  identifies paste (such as bracketed paste). Even a trailing pasted newline
  waits for Enter. Terminals that send paste as ordinary keystrokes cannot offer
  this guarantee: pasted newlines may submit partial prompts.

Interactive prompts retain their whitespace. Input history holds whole submitted
prompts and slash commands in memory for the current process, including prompts
whose generation failed or was interrupted. Consecutive duplicates share one
history entry. Input recall history is not saved to disk. `/clear` clears conversation context,
not input recall history; `/history` displays conversation context.

**Ctrl+C** at the prompt discards the draft and exits. During generation it aborts
the response and returns to the prompt, rolling back the incomplete conversation
turn. **Ctrl+D** exits on an empty buffer; with text present it deletes the
character under the cursor. On Windows, **Ctrl+Z** also exits immediately on an
empty buffer and is ignored with text present. Input closure discards an
unsubmitted draft and exits. The editor restores terminal modes before returning.

The editor uses `prompt_toolkit` for Windows, macOS, and Linux terminal support.
Shortcut delivery depends on terminal configuration. When stdin or stdout is
redirected, lclaude retains its simple line-oriented input: each line is a separate
submission with outer whitespace trimmed, and interactive editing is disabled.


## Context display

Interactive sessions show a fixed status bar at the bottom, updated as the
conversation grows or the selected model changes. It stays below streamed output
and does not add status lines to the transcript. When the runtime context size is
unknown, the bar is hidden. Redirected output has no automatic context display.
`/context` explicitly shows instructions (or the default prompt), conversation
content, formatting overhead, and usage reported for the last completed request.

```sh
lclaude --num-ctx 8192 --max-response-tokens 2048
```

Explicit `--num-ctx` is passed to Ollama. Otherwise the display uses the selected
model's loaded `context_length` from `/api/ps`, never its architectural maximum.
Context discovery errors and oversized estimates do not block chat. lclaude does
not preload a model just to populate the display or remove conversation messages.
Ollama controls what happens when the actual context allocation is exhausted.

Counts prefixed with `~` are approximate: content is estimated at three UTF-8
bytes per token, plus 16 tokens per message and 16 for reply framing. This heuristic
is replaceable and is not an exact tokenizer or a fit guarantee. Ollama's reported
`prompt_eval_count` and `eval_count` appear separately in `/context`, including a
comparison with the estimate for that completed request.

The reply limit defaults to 2,048 tokens and is passed as `num_predict`. Neither
the estimate nor an unknown context size prevents a response.


## Saved chats and resume

Completed turns are saved automatically under
`~/.local_claude/chats/<project-hash>/<session-UUID>.json`. The project hash is
SHA-256 of the resolved absolute launch-directory path; separate launch directories
have separate catalogs, even if they share a directory name. Each new session gets
a UUID4. Files use schema version 1 and contain the project path, timestamps with
timezones, selected model, and full messages. Project instructions are not stored.
No token estimates, display state, or shortened messages are stored.

Use `/chat` inside lclaude to browse saved chats for the current project, ordered
from most recently updated to least recently updated. The scrollable picker shows
each chat's local date and time (with AM/PM) and first-message preview, and marks
the current chat. **Up/Down** move between chats, **Page Up/Page Down** move ten
rows, **Enter** resumes the highlighted chat, and **Escape**, **Ctrl+C**, or
**Ctrl+D** cancel. With redirected input, `/chat` prints the list without consuming
another line of input; selection requires an interactive terminal.

Resuming restores the saved model (which must still be installed) and uses the
**current instructions loaded at launch**. Earlier draft files containing `system_prompt` remain readable; that field is
ignored and removed on the next save. Restarting
lclaude loads the current project file for both new and resumed chats. Changes
made while lclaude is running take effect on the next launch.
The saved conversation is rendered before the next prompt, with user messages and
Markdown assistant responses available in terminal scrollback. Use `/model` after
resuming to change models.
If the selected chat cannot be read or its model is unavailable, the current chat
remains active. The current chat is saved before switching; a save failure is silent
and does not block the switch.

`/clear` preserves instructions and the selected model and starts a fresh chat with
a new UUID. The original chat remains available in `/chat`. The new chat is saved
only after its first completed user/assistant turn; empty chats and interrupted
first turns create no files. Changing models in an empty chat also creates no file.
`/clear` starts a new chat even if saving the old chat fails.
Launching lclaude automatically resumes the most recently updated saved chat for
the launch directory and displays its transcript. If no valid saved chat exists,
it starts fresh. Invalid files are reported and skipped. The saved model is used
unless `--model` explicitly overrides it; an unavailable model produces a startup
error rather than silently switching models. Use `/chat` to open another chat.
Because empty chats are not saved, exiting immediately after `/clear` reopens
the most recent saved chat on the next launch. Complete a turn in the new chat
to make it the most recent saved conversation.
Browsing or clearing an unchanged chat does not update its saved activity time.

Failed or interrupted generation does not save the pending turn. Each save writes
a temporary file beside the destination, flushes and fsyncs it, then atomically
replaces the JSON file. A write failure is silent; the in-memory conversation
remains usable and a later successful save includes it.
Unsaved changes are lost on exit. Malformed or unsupported files are reported and
never intentionally overwritten. Chats are plain local JSON, so their full text
remains on disk after `/clear`. Avoid resuming the same UUID in multiple processes;
concurrent writers are not merged or locked in this initial implementation.
