# lclaude

[![CI](https://github.com/AkshayKamath12/lclaude/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AkshayKamath12/lclaude/actions/workflows/ci.yml)

**Chat and inspect your project with local Ollama models from your terminal.** lclaude streams responses, loads project instructions, and saves chats so you can resume them later. It runs on Windows, macOS, and Linux with Python 3.10 or newer; no hosted AI account is needed with the default local setup.

> lclaude provides chat, an approval-gated command tool, model selection, context budgeting, and saved sessions. Commands run with your permissions and can modify files; the working directory is not a sandbox.

## Install

Install [Python 3.10+](https://www.python.org/downloads/), [Git](https://git-scm.com/downloads), and [Ollama](https://ollama.com/download). For the first option below, install [uv](https://docs.astral.sh/uv/getting-started/installation/). Make sure the Ollama server is running and at least one model is installed.

**With uv** (installs lclaude as an isolated command-line tool):

```sh
uv tool install git+https://github.com/AkshayKamath12/lclaude.git
```

**With pip** (run inside the Python environment where you want lclaude installed):

```sh
python -m pip install git+https://github.com/AkshayKamath12/lclaude.git
```

## Launch

```sh
ollama pull qwen2.5:7b-instruct  # optional example model
lclaude
```

By default, lclaude uses the most recently saved chat's model. For a new chat it tries `qwen2.5:7b-instruct`, then falls back to the first installed model. A saved chat's model must still be installed; use `--model` to override it.

### CLI flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `-h`, `--help` | — | Show help and exit. |
| `-m`, `--model NAME` | Saved chat model, or `qwen2.5:7b-instruct` | Choose an installed Ollama model. For a new chat, lclaude falls back to the first installed model if the default is unavailable. |
| `--host URL` | `http://localhost:11434` | Ollama server address. |
| `--timeout SECONDS` | `60` | Ollama client timeout. |
| `--num-ctx TOKENS` | Discover at runtime | Set Ollama's context allocation. |
| `--max-response-tokens TOKENS` | `2048` | Maximum generated response size. |
| `--max-tool-iterations COUNT` | `10` | Maximum model responses per user turn, including tool requests. |

### Slash commands

| Command | Purpose |
| --- | --- |
| `/chat` | Browse and resume saved chats for the current launch directory. |
| `/clear` | Start a new chat while keeping earlier saved chats. |
| `/context` | Show estimated prompt usage and the last reported token counts. |
| `/history` | Preview messages in the active chat. |
| `/model [NAME]` | Open the model picker or switch directly to `NAME`. |
| `/help` | List available commands. |
| `/exit` | Exit lclaude. |

## Project behavior

- At launch, lclaude loads `AGENTS.md` from the current directory, or `agents.md` if the uppercase file is absent. A blank or missing file uses a default prompt. Instructions are loaded again on the next launch.
- Completed chats are stored as JSON under `~/.local_claude/chats/`, grouped by launch directory. Startup resumes the newest valid chat; `/clear` starts a fresh one.
- In interactive mode, **Enter** submits and **Alt+Enter** inserts a newline. **Ctrl+C** exits at the prompt or aborts the current response. Redirected input is handled one line at a time.
- The context footer estimates instructions, conversation, tool definitions, and reserved reply space. Every model request must fit the estimated budget. If it cannot fit, lclaude explains how to adjust the context and preserves recorded command activity; it never silently removes messages.

## Command tool

With an Ollama model that supports structured tool calling, the model can request
`run_command`. You see the full command, shell, directory, and timeout before
approving each call. Windows uses Windows PowerShell; Linux/macOS use `/bin/sh`.
Relative directories resolve against the launch directory. Redirected input
cannot approve commands. Ctrl+C cancels execution; it cannot undo effects.

Stdout and stderr each retain at most 6,000 bytes, with an explicit truncation
notice; omitted output is not saved. Completed results survive interrupted
responses. On resume, lclaude explains pending/unknown outcomes and waits for
your next message without rerunning commands. Version-1 and version-2 chats remain
readable. See [tool behavior and session format](docs/tool-loop.md) for limits,
approval, persistence, and interruption details.

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
```
