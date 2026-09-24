# Load project instructions at startup

Status: implemented; automated checks passed on 2026-09-23.
Related to [issue #16, part 1](https://github.com/AkshayKamath12/lclaude/issues/16)
and the [HLD](../hld.md).

## Goal

When a user launches `lclaude` in a project directory, read its `AGENTS.md`
and include those instructions in every model request. This gives the model
project guidance without the user pasting it into chat.

Before this change, `Session` already accepted a system prompt, prepended it to
requests, and preserved it through `/clear`. The application called `Session()`
without supplying that prompt. This change loads the file and connects these pieces.

## Behavior

- Look in the launch directory only: try `AGENTS.md`, then `agents.md` if the
  first is absent. If both exist as separate files, uppercase wins. On a
  case-insensitive filesystem, the same file is read once.
- Read once at startup using UTF-8 with optional BOM support (`utf-8-sig`).
  Preserve the Markdown text; do not parse it or follow referenced files.
- Use the file contents as the system prompt. If neither file exists, or the
  selected file contains only whitespace, use this default:
  `You are a helpful coding assistant.`
- If the selected path is a directory, unreadable, or invalid UTF-8, report its
  path and the reason on stderr and exit with status 1 before starting input.
  Do not silently fall back to another file after a read error.
- Keep the loaded prompt for the whole session, including after `/clear`,
  failed turns, and model changes. Restart lclaude to pick up file edits.
- Apply the same loading behavior to interactive and redirected input.

For example, launching in a folder whose `AGENTS.md` says `Use pytest for tests`
produces requests in this order:

```text
system: Use pytest for tests
user: How should I test this function?
```

This supplies guidance to the model; it does not execute instructions or
guarantee that the model follows them.

## Implementation steps

1. Add `src/lclaude/instructions.py` with
   `load_system_prompt(directory: Path) -> str`, the default prompt constant,
   and an `InstructionLoadError` that identifies the failing path. Keep file
   lookup and decoding here; this module does not print or call Ollama.
2. In `cli.main()`, capture `Path.cwd()` and load the prompt after argument
   parsing, before engine startup. Handle lookup/read errors as startup errors
   on stderr. Pass the resulting string into `run_chat_loop` through a
   keyword-only `system_prompt` argument.
3. Have `run_chat_loop` create `Session(system_prompt=system_prompt)`. Keep its
   argument default as `None` for existing direct callers. `Session` continues
   to own message ordering; the inference engine needs no changes.
4. Add a short README section documenting filenames, launch-directory lookup,
   fallback behavior, and the need to restart after editing instructions.

## Tests and completion

- Loader tests use temporary directories: uppercase file, lowercase fallback,
  both filenames where supported, missing/blank file, Unicode/BOM, and a parent
  file that must not be loaded. Mock permission failures for portability; also
  cover a directory at the selected path and invalid UTF-8.
- Startup tests verify the launch directory is used, the loaded/default prompt
  reaches the loop, and a load failure exits before input or inference. Isolate
  these tests from the repository's actual `AGENTS.md`.
- Loop tests inspect requests across multiple turns, `/clear`, interruption,
  and model selection: exactly one system message remains at index zero, and
  failed turns still roll back. Existing session tests already cover system
  prompt prepending and preservation on clear.
- Run focused loader/CLI/session/model-selection tests during implementation,
  then `pytest`, `ruff check .`, `mypy src`, and `python -m build`.

Part 1 is complete when the request payload consistently contains the selected
instructions and missing files still allow normal chat.

Verification completed: focused tests passed (71 passed, 1 skipped); the full
suite passed after updating to current main (186 passed, 1 skipped), as did
`ruff check .`, `mypy src`, and
`python -m build`, using the project virtual environment. The skipped test needs
two filenames differing only in case, which this Windows filesystem does not
support. Mocked Ollama requests verify payloads; no live-model trial was run.

## Later work

Token counting and oversized-prompt rejection belong to part 2 of #16. This
step loads the full file without truncation and does not yet enforce a context
budget. Parent/nested discovery, global instruction files, live reload, skills,
and summarization are outside this change.
