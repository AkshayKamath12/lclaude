# lclaude — Agent Instructions

lclaude is a terminal developer tool for local models through Ollama. Build it as a dependable tool for daily use, with particular attention to slow inference, limited context, and interrupted sessions.

Read `docs/hld.md` for the project’s long-term direction. Distinguish planned features from implemented behavior by inspecting the current code and tests.

## Architecture

Keep these responsibilities distinct, even if their files or package structure change:

- **Inference engine:** Communicates with Ollama, streams responses, and translates transport and model failures into errors the rest of the application can handle. It does not render terminal output or decide how conversation state is stored.
- **Session state:** Owns conversation history and turn lifecycle. Changes to a turn must be atomic: commit a complete response or roll back after failure or interruption.
- **Command handling:** Recognizes and executes user-facing slash commands. Keep command behavior separate from ordinary model responses.
- **Terminal UI:** Owns input editing, colors, progress indicators, streamed output, and terminal cleanup. UI improvements must not change session semantics.
- **Application loop:** Coordinates input, commands, inference, session state, and UI. Keep orchestration here rather than letting the other components depend on one another unnecessarily.

Preserve these boundaries when adding features, but use judgment when evolving the structure. Do not keep a poor design solely to match this description.

## Project Goals

These goals describe the long-term direction of lclaude. They should guide design decisions, but they are not all requirements for the current task. Implement only the requested scope unless broader work is explicitly necessary.

* **Build a production-grade local developer tool.** lclaude should be more than a thin wrapper around an Ollama request. It should remain dependable during slow inference, connection failures, exhausted resources, interrupted streams, and long-running sessions.

* **Provide a polished terminal experience.** The interface should feel responsive, intentional, and comparable to mature tools such as Claude Code. Long-term UI goals include multiline editing, persistent command history, slash-command autocomplete, model selection, readable streamed Markdown, syntax-highlighted code, clear thinking and tool-execution states, latency and throughput information, context-window visibility, and reliable terminal restoration after interruptions.

* **Manage limited context deliberately.** Local context windows and memory are constrained resources. Track token usage, reserve space for generation, and assemble prompts according to explicit budgets rather than relying on unbounded history or naive FIFO truncation. Preserve essential system instructions, repository guidance, explicit user corrections, the active objective, and recent dialogue. Compact or summarize older context without breaking tool-call relationships or silently losing important decisions.

* **Support durable, inspectable sessions.** Conversations should be resumable and auditable. Persist complete session data before compacting its active representation, and ensure failed or interrupted turns cannot corrupt saved history.

* **Evolve into a safe coding agent.** Move beyond chat toward repository inspection, file references, command execution, patch application, and iterative verification. Actions that can modify the workspace or execute commands should be transparent and pass through an interactive approval boundary unless the user has explicitly configured a trusted workflow.

* **Remain Unix-native and composable.** Support interactive use as well as pipelines, redirected input, and headless commands. The tool should work naturally with shell commands, logs, diffs, scripts, and CI environments without requiring the interactive UI.

* **Stay local-first and privacy-conscious.** Core inference and project interaction should work through local models without requiring source code to be sent to a hosted model. Optional external integrations must be deliberate and visible to the user.

* **Scale across local hardware.** Adapt model selection, context limits, and memory budgets to different hardware envelopes. Avoid assumptions of unlimited RAM, VRAM, compute, or inference speed.

* **Keep the system modular and understandable.** Maintain clear conceptual boundaries between architectural boundaries mentioned above. The concrete package structure may evolve, but responsibilities should remain testable and loosely coupled. Prefer transparent components over heavy orchestration frameworks.

* **Make extension straightforward.** Repository instructions, skills, context resources, and tools should be discoverable on demand through progressive disclosure. Adding a capability should not require rewriting the core agent loop.

* **Treat quality automation as part of the product.** Maintain fast unit tests and meaningful coverage of failure paths, especially streaming, interruption, rollback, persistence, and context compaction. CI should automatically enforce tests, linting, type checking, packaging integrity, and supported-platform behavior. Release automation should eventually produce reproducible, versioned builds with clear changelogs.

* **Measure behavior instead of guessing.** Surface and test useful operational information such as time to first token, generation throughput, token consumption, context utilization, and failure causes. Performance work should be based on observable bottlenecks.

* **Remain cross-platform and easy to install.** Preserve consistent behavior across macOS, Linux, and Windows while respecting terminal and filesystem differences. Distribution should use standard Python packaging and provide a predictable upgrade path.


## How to work on this project

- Inspect the relevant code and tests before changing behavior.
- Implement the requested feature without pulling unrelated features forward from the long-term design.
- Prefer small, understandable components over a heavy orchestration framework.
- Test meaningful success and failure paths, especially for streaming, interruption, and state changes.
- Run the relevant project checks. Report what ran, what changed, and any remaining limitation. Do not claim a check passed unless you ran it.

## Git and GitHub workflow

- Follow `CONTRIBUTING.md` for branch, commit, PR, and issue conventions.
- For preparing, opening, or updating PRs, read `.agents/skills/raise-pr/SKILL.md`.
- Always create PRs as drafts (`gh pr create --draft`). Mark ready only when explicitly requested.
- Prefer a single commit per PR and squash merges into `main`; use a commit message describing the final scope. Rewriting a published branch still requires authorization, and must use `--force-with-lease`.
- Preserve unrelated working-tree and staged changes. Commit only the intended task scope.
- A request to open or raise a PR authorizes committing and pushing the intended changes and creating the PR. A request to prepare a PR is local-only unless publishing is also requested.
- Merging, publishing releases, and rewriting shared history require explicit authorization; honor authorization already given in the session.

## Verification

Run the checks relevant to the change:

- Tests: `pytest`
- Lint: `ruff check .`
- Type checking: `mypy src`
- Packaging: `python -m build`

Prefer the smallest relevant test set while iterating, then run the broader checks before declaring the task complete. If a check cannot run, explain why. Do not claim that behavior is verified without running the corresponding check.
