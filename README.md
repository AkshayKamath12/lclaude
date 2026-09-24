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
after editing the file. This also applies with redirected input. Files are loaded
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
history entry. Nothing is saved to disk. `/clear` clears conversation context,
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
