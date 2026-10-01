# lclaude

[![CI](https://github.com/AkshayKamath12/lclaude/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AkshayKamath12/lclaude/actions/workflows/ci.yml)

**Chat and inspect your project with local Ollama models from your terminal.** lclaude streams responses, loads project instructions, and saves chats so you can resume them later. It runs on Windows, macOS, and Linux with Python 3.10 or newer; no hosted AI account is needed with the default local setup.

> lclaude provides chat, read-only project tools, model selection, context estimates, and saved sessions. It does not yet edit files or run shell commands.

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
- The context footer estimates instructions, conversation, tool definitions, and reserved reply space. Plain chat keeps its advisory budget. For tool conversations, lclaude reduces retrievable excerpts to fit; if complete call/result pairs still cannot fit, it stops and keeps the saved transcript.

## Project tools

With an Ollama model that supports structured tool calling, the model can use
`list_files`, `read_file`, `search_text`, and `read_tool_output`. Paths resolve
against the launch directory. Tool activity appears in the terminal, and large
outputs remain available as session-owned artifacts.

Completed tool results survive interrupted responses. On resume, lclaude shows
the saved work and waits for your next message. Version-1 chats remain readable.
See [tool behavior and session format](docs/tool-loop.md) for limits, retrieval,
and interruption details.

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
```
