# High-Level Design Document: lclaude

**Status:** Draft   
**Target Systems:** Cross-Platform (Windows, macOS, Linux)  
**Primary Dependencies:** Python 3.10+, Ollama, pathlib  
**Distribution:** Global CLI installed via `uv tool` or `pipx`

---
## Overview

### The Problem
Modern developers face a dilemma when integrating AI into their local workflows:
1. **Cloud Dependency:** Cloud-based assistants (Claude, Copilot, Cursor) require sending your source code to external servers and paying their subscriptions.
3. **Local Model Limitations:** Local models struggle with managing memory. Users need to manually feed in fixed windows of context and summarize file reads and logs to avoid memory bloat. Local models are also stateless with regards to your file system. They have no context for the current directory or project. Also, your chats are not persisted. 


### The Solution
**Local Claude** is like the Claude Code CLI interface, but for local models. It is a terminal-native AI wrapper that orchestrates local inference (via Ollama) without requiring any cloud connections or subscriptions.

It includes core capabilities that Claude has for managing chats, selecting models, and managing chat memory. By using token budgeting and file compaction, it allows developers to maintain long-running coding sessions without crashing their machine's memory limit.

### The High-Level Idea
The core philosophy of this project is serving as a minimalistic wrapper. The CLI itself ships with zero domain knowledge and workflow assumptions. 

Instead, it serves as a harness that knows how to execute. It manages the token math, terminal streaming, file I/O, and model tool-calling. The actual intelligence is injected dynamically at runtime:
* **Repository Awareness:** When launched, the CLI reads a local `agents.md` file in the current directory, instantly adopting the tech stack, rules, and persona of that specific codebase.
* **Progressive Disclosure:** Instead of memorizing documentation, the harness maintains a library of domain skills (markdown playbooks). It only loads a specific skill or context file into active memory when relevant to the user's prompt.
* **Hardware Elasticity:** The harness scales its memory limits dynamically by reading a `config.json` file.

---

## System Architecture

The harness coordinates state, model communication, and local execution across five subsystems.

~~~text
┌────────────────────────────────────────────────────────────────────────┐
│                        Terminal User Interface                         │
│   (Interactive prompt, command router, streaming stdout, error traps)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Context & Memory Engine                          │
│  - System Pinned Layer (agents.md + system_behavior.md + Catalog)      │
│  - Conversation Ledger (Rolling objectives & status)                   │
│  - Sliding Window (Dialogue token budget manager)                      │
│  - Ephemeral Tool Compactor (Converts large dumps to receipts)         │
└───────┬───────────────────────────┬────────────────────────────┬───────┘
        │                           │                            │
        ▼                           ▼                            ▼
┌───────────────┐           ┌───────────────┐            ┌───────────────┐
│ Dynamic Skill │           │ Storage Layer │            │  Inference    │
│    Registry   │           │ (Disk JSON)   │            │   Adapter     │
│ - Global      │           │ - Audit Logs  │            │ - Ollama API  │
│ - Local Repo  │           │ - Search      │            │ - Model List  │
│ - On-Demand   │           │ - Resume      │            │ - Tool Calls  │
└───────────────┘           └───────────────┘            └───────────────┘
~~~

### Core Subsystems

1. **CLI** Manages terminal sessions, catches user interrupts, parses slash commands (`/model`, `/search`, `/load`), and handles streaming responses.
2. **Context & Memory Engine:** Assembles prompt payloads, balances the active token budget, prevents orphaned tool messages, and collapses verbose historical outputs into concise receipts.
3. **Inference Adapter:** Interfaces with Ollama over HTTP via the official Python SDK, executing recursive function-calling loops when tool calls are emitted.
4. **Dynamic Skill Registry:** Discovers markdown playbooks, extracts shallow trigger metadata for the base prompt, and exposes the retrieval tools (`load_skill`, `load_context`).
5. **Storage & Session Engine:** Writes conversation turns synchronously to project-namespaced JSON files, providing historical search, resumption, and full auditability.

---

## Workspace Hierarchy: Global vs. Local State

To allow a single global CLI to operate across multiple repositories without workspace pollution, the filesystem is partitioned into two distinct scopes. Paths are resolved using Python's `pathlib` for native macOS and Windows compatibility.

~~~text
~ [User Home Directory]
└── .local_claude/                      <-- Global configuration/state
    ├── config.json                     <-- Hardware budgets and model defaults
    ├── system_behavior.md              <-- Immutable operating protocols including progressive disclosure
    ├── skills/                         <-- Cross-project reusable playbooks for how to do things
    └── chats/                          <-- Centralized session persistence in JSON files

[Active Project Directory]
├── agents.md                           <-- Local repository architecture & rules
├── .agent/                             <-- Project-specific overrides (Optional)
│   ├── skills/                         <-- Repo-specific operational playbooks
│   └── context/                        <-- Repo-specific reference specs/schemas
└── src/                                <-- Application source code
~~~

### Resolution Rules
* **Configuration:** Merges `~/.local_claude/config.json` with `./.agent/config.json` (local overrides global).
* **Workspace Context:** If `./agents.md` exists in the working directory, it is loaded. If absent, the engine falls back to an agnostic generic assistant persona.
* **Skills Catalog:** The engine scans `~/.local_claude/skills/` first, then `./.agent/skills/`. Local playbooks override global playbooks if filenames collide.

---

## Memory Architecture

To prevent inference degradation, the payload sent to Ollama on each turn is assembled using three distinct layers.

~~~text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Pinned System Layer (Always sent at index 0, never evicted)         │
│    - Global operational protocols (system_behavior.md)                 │
│    - Shallow catalog of available skills (metadata triggers only)      │
│    - Active workspace context (agents.md)                              │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Conversation Ledger (Pinned session memory)                         │
│    - High-level goal of current session                                │
│    - Established technical decisions                                   │
│    - Active task status                                                │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Sliding Dialogue Window (Bounded by dynamic token budget)           │
│    - Turn k-2: User input                                              │
│    - Turn k-1: Assistant tool call + Compacted tool receipt            │
│    - Turn k:   Current prompt                                          │
└────────────────────────────────────────────────────────────────────────┘
~~~

### Dynamic Token Budgeting Across Hardware Profiles

Token budgets are driven by configuration values in `config.json`, allowing the same engine to scale across varying RAM envelopes without code changes:

| Profile | Hardware Envelope | `num_ctx` | Dialogue Budget | Model Scale |
| :--- | :--- | :--- | :--- | :--- |
| **Constrained** | 16 GB Unified (Apple M4) | 8,192 | ~4,500 tokens | 7B – 9B |
| **Expanded** | 32 GB RAM (Windows/MSI) | 16,384 | ~10,000 tokens | 14B – 32B |

### Compaction
Tools that return large blobs of text (e.g., file readers, linter outputs) cause rapid context exhaustion if kept in memory indefinitely.
1. **Synchronous Disk Write:** The full, unaltered tool output is appended to the session `.json` file on disk for a permanent audit trail.
2. **Active Context Compaction:** On subsequent turns, the memory builder inspects the dialogue slice. Historical tool outputs exceeding `max_tool_chars` are truncated to a compact receipt in the active payload (e.g., `[Tool output for read_file("schema.sql") compacted. Full output preserved on disk.]`).
3. **Pair Integrity:** The slicing algorithm guarantees an assistant tool-call message is never separated from its subsequent tool-receipt message.

---

## Progressive Disclosure (The Skill Engine)

To prevent saturating the context window, the system avoids loading vast documentation into the base prompt. 

1. **Shallow Cataloging:** On boot, the wrapper reads only the first line (the trigger description) of all `.md` files in the `skills/` directories, compiling a lightweight catalog for the base prompt.
2. **Intent Matching:** If a user requests a complex workflow, the model reads the catalog and emits a structured JSON tool call: `load_skill("specific_skill")`.
3. **Deep Injection:** The wrapper reads the deep operational checklist from disk and feeds it back as a `tool` response.
4. **Execution:** The model generates the final response adhering strictly to the newly injected rules.

---

## Dynamic Tool Registry (Bring-Your-Own-Tools)

The harness supports an auto-registering Python tool system. New system capabilities are added by writing standard Python functions in `tools.py` with descriptive type hints and docstrings. 

On boot, the harness dynamically inspects these functions and translates them into Ollama's expected JSON Schema format. This allows developers to easily extend the CLI (e.g., adding `query_postgres` or `run_pytest`) without modifying the core execution loop.

---

## Distribution & Installation

The CLI is distributed as a standard Python package, avoiding complex `.sh` or `.bat` installation scripts. It relies on standard package managers to create isolated virtual environments and cross-platform executable binaries.

**Configuration (`pyproject.toml`):**
~~~toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "local-claude"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["ollama>=0.3.0"]

[project.scripts]
lclaude = "local_claude.cli:main"
~~~

**Installation Command:**
~~~bash
uv tool install git+https://github.com/<username>/local-claude.git
# or
pipx install git+https://github.com/<username>/local-claude.git
~~~

This single command automatically resolves OS differences, building `lclaude` in `~/.local/bin` (macOS/Linux) or `%USERPROFILE%\AppData\Local\Programs` (Windows).