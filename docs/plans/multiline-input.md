# Multiline input implementation plan

Status: implemented; automated verification complete, with pre-existing lint/type
failures recorded below. Native terminal smoke tests not run. Updated: 2026-09-23.

This is a living plan. Record review decisions and implementation discoveries here
before changing the agreed behavior. Only this document is changed during planning.

## Current behavior and architecture

Inspected `AGENTS.md`, `CONTRIBUTING.md`, `docs/hld.md`, `README.md`,
`pyproject.toml`, `src/lclaude/{ui,cli,commands,session}.py`, the corresponding
four test modules, and `.github/workflows/ci.yml`.

- `ui.get_user_input()` calls `input("\n> ").strip()`. It returns `None` on
  `KeyboardInterrupt` or `EOFError`. There is no application-managed editor,
  input history, multiline composition, or bracketed-paste configuration.
  Incidental editing provided by a particular Python terminal is not a contract.
- `cli.run_chat_loop()` owns one conversation `Session`. It ignores empty input,
  routes strings starting with `/` to command handling, and otherwise appends a
  user message before streaming. It appends the assistant response only after
  success, and rolls back the pending user message on stream interruption or
  handled engine errors. This lifecycle must remain unchanged.
- Prompt interruption/EOF prints `Session terminated by user.` and exits the
  loop; it also sets SIGINT to SIG_IGN to protect shutdown. Generation Ctrl+C
  prints an abort notice and returns to input after rollback.
- Commands match the entire stripped, lowercased string. `/clear`, `/history`,
  `/help`, and `/exit` exist. `/history` shows conversation previews, not editor
  history; `/clear` clears conversation state. `/exit` raises `SystemExit(0)`.
  A multiline string beginning with `/` is currently handled as one command,
  not split into multiple commands. Preserve this rule.
- Output is synchronous stdout/stderr writing. Input editing and streamed output
  do not run concurrently. No full-screen UI or asynchronous application-loop
  migration is needed.
- Existing UI tests mock `builtins.input`, including trimming and interruption.
  Loop tests cover success, stream interruption, connection rollback, and prompt
  shutdown. Session tests cover snapshots, rollback, clear, and newline previews.
  There are no actual editor/key-sequence tests yet.
- The HLD describes future persistence, context budgeting, tools, and richer UI;
  those are not implemented requirements for this change.

## Library and dependencies

Recommend a direct runtime dependency `prompt-toolkit>=3.0.52,<4` in
`pyproject.toml`, subject to installation and compatibility verification after
approval. Version 3.0.52 is the documentation/source baseline inspected, not a
claim that it is the latest release. Retain Python 3.10 support.

The active interpreter is `.venv/Scripts/python.exe`. Both installed distribution
metadata and `importlib.util.find_spec('prompt_toolkit')` show it is absent.
It is not declared in `pyproject.toml`; no repository lockfile was found. Nothing
was installed during planning.

The library supplies multiline buffers, terminal backends, memory history, paste
handling, and configurable key bindings. It supports Windows and Unix-style
platforms, making it preferable to implementing raw input and terminal restoration
ourselves. See the [upstream project](https://github.com/prompt-toolkit/python-prompt-toolkit)
and [package metadata](https://pypi.org/project/prompt_toolkit/).

Alternatives: stdlib `input` cannot provide this editing contract; readline is
not a consistent cross-platform multiline abstraction; a full-screen framework
would expand scope. Accept the additional runtime library and its transitive
dependencies (including terminal-width handling) in exchange for maintained
terminal support. Verify resolved dependency metadata during implementation.

## Proposed UI ownership and components

| Responsibility | Current path | Planned change |
| --- | --- | --- |
| Input editing and terminal lifecycle | `src/lclaude/ui.py` | Introduce a small UI-owned input reader holding one `PromptSession[str]` and `InMemoryHistory`; provide a `read() -> str | None` boundary, replacing the stateless input function or making it a thin adapter. |
| Application orchestration | `src/lclaude/cli.py` | Construct the reader once per chat loop and reuse it; keep command dispatch, inference, and session mutation here. |
| Input guidance | `src/lclaude/ui.py`, `README.md` | Document submission, newline fallback, history, exits, and paste limitations. |
| Dependency declaration | `pyproject.toml` | Add the justified runtime dependency. |
| Input and loop verification | `tests/test_input.py`, `tests/test_ui.py`, `tests/test_cli.py` | Test editor events independently and adapt loop mocks to the UI boundary. |
| Conversation and command invariants | `tests/test_session.py`, `tests/test_commands.py` | Retain existing coverage; add cases only where multiline integration exposes a meaningful gap. |

Prefer keeping the small editor in `ui.py`; split into a UI input module only if
its size warrants it. No engine or session production changes are expected.
The reader must not import Ollama or mutate conversation state. Explicit instance
ownership avoids global history leaking between tests or separate loop invocations.
One chat loop is the current CLI process lifetime; no new process-wide singleton.

Use a synchronous `PromptSession`, multiline enabled, ordinary Emacs editing,
`> ` for the first line and `... ` for continuation lines. Keep ordinary wrapping.
No completer, suggestions, syntax highlighting, model picker, external editor,
file history, or output-rendering changes. Let each prompt finish and restore
terminal state before commands or streaming print output.

## Proposed keys and text handling

| Input | Proposed behavior |
| --- | --- |
| Enter | Accept the complete nonblank buffer as one prompt. |
| Alt+Enter | Insert one literal newline at the cursor, without automatic indentation or submission. |
| Escape, then Enter | Portable fallback for newline insertion; press sequentially as one key sequence. |
| Up / Down | Move inside the buffer first; navigate history at the first/last logical line when no further logical vertical movement is possible. |
| Left / Right, Home / End | Normal library editing behavior. |
| Ctrl+C at prompt | Discard draft and exit the chat, even with text present, preserving current behavior. |
| Ctrl+D on empty buffer | EOF: exit the chat. On nonempty input, retain ordinary forward-delete behavior, never submit. |
| Windows Ctrl+Z on empty buffer | Proposed explicit EOF alias; avoid the library inserting a literal control character. On nonempty input, ignore rather than submit or discard text. |
| Actual input closure/EOF | Discard any unsubmitted draft and return `None`. |

Override the library's default multiline acceptance keys: its usual Enter inserts
a newline, so merely enabling multiline is insufficient. Bind `enter` to accept
and `escape`, `enter` to insert a newline. Retain LF-as-Enter compatibility;
do not use Ctrl+J as the advertised fallback because some terminals send LF for
Enter. The Escape sequence avoids depending on a distinct Shift+Enter encoding.
Alt/Option delivery can be intercepted or configured by the terminal; cannot
guarantee Alt+Enter everywhere. Library documentation explains the Escape prefix
and sequence timeout behavior: [key bindings](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html).

Approved interactive whitespace policy: use `text.strip()` only to detect blank
input; preserve the actual accepted text, including indentation, boundary spaces,
and leading/trailing newlines. This deliberately changes existing whole-input
trimming so pasted code is not damaged. Detect commands with
`user_input.lstrip().startswith('/')`; the existing command handler still strips
for matching. Thus padded slash commands continue to work. Blank Enter stays in
the editor and records nothing; the loop retains its defensive empty-input guard.

## Paste and compatibility limits

Keep the library's bracketed-paste handler: insert the payload as text, preserving
line boundaries and indentation, normalizing CRLF/CR to LF. Embedded newlines
must not invoke acceptance. Even a paste ending in a newline waits for a separate
Enter. Do not split pasted content into commands or prompts. Upstream implementation:
[basic bindings](https://raw.githubusercontent.com/prompt-toolkit/python-prompt-toolkit/3.0.52/src/prompt_toolkit/key_binding/bindings/basic.py).

Without bracketed paste or another distinguishable native paste event, pasted
newlines can be indistinguishable from typed Enter. There is no reliable generic
solution while Enter always submits. Recommend guaranteeing safe multiline paste
where the terminal/backend identifies paste and documenting other terminals as a
limitation. Do not use timing heuristics. If universal unmarked-paste protection
is required, an explicit paste mode is additional scope that needs approval.

Manual compatibility checks should cover Windows Terminal/PowerShell, macOS
Terminal or iTerm2, and a Linux VT-compatible terminal, including terminal resize,
wrapped lines, Unicode width, paste, and return to the shell after interruption.
Terminal shortcut configuration may be needed for Alt/Option+Enter; the fallback
must work without custom terminal mappings. Do not promise arbitrary terminal
emulators or remappings work. Mocked backend tests do not establish native-terminal
compatibility; report which platforms were actually exercised.

For redirected stdin/stdout, recommend retaining the existing line-oriented
`input()` path when either stream is not a TTY. This avoids terminal escape output
and preserves existing pipeline behavior. It does not add headless multiline
framing: each input line remains a separate submission. Interactive multiline
guarantees apply to the editor path. Test this selection explicitly and allow
injected editor I/O to bypass TTY detection in unit tests.

## History and cursor movement

Use one in-memory history for the active reader, separate from conversation
`Session`. Save complete nonblank submissions, including slash commands and
prompts whose later inference fails or is interrupted. `/clear` must not erase
editor history; `/history` continues to describe only conversation context.
Do not save cancelled drafts. No files, persistence, or cross-session restoration.

Recommend chronological navigation with prefix search disabled. Allow the library's
consecutive-identical-entry suppression; distinct multiline submissions remain
intact. Recall produces an editable buffer, not an immediate submission. Returning
past the newest history item restores the draft being edited. Editing recalled
text must not rewrite the stored entry.

Recommended boundary interpretation is first/last *logical line*, not absolute
character offset zero/length. Retain the desired column when moving through lines
of different lengths. Soft wrapping is presentation: a wrapped logical line still
counts as one line for this initial scope. This can cause Up on a visually wrapped
first line to enter history; the user approved this tradeoff. Visual-row
navigation would require additional layout-aware handling and resize tests.
Verify the library's `auto_up`/`auto_down` and draft restoration rather than assume
all desired behavior from configuration alone: [buffer source](https://raw.githubusercontent.com/prompt-toolkit/python-prompt-toolkit/3.0.52/src/prompt_toolkit/buffer.py).

## Interruption, EOF, and cleanup

Catch `KeyboardInterrupt` and `EOFError` only at the UI read boundary and return
`None` after the library has unwound its terminal context. Preserve loop exit
messages, shutdown SIGINT handling, `/exit`, and generation rollback. Do not
convert prompt Ctrl+C into merely clearing the buffer.

The library owns raw-mode entry/exit, cursor visibility, and bracketed-paste
enable/disable. No active prompt remains during model streaming. Test restoration
on accepted input, Ctrl+C, EOF, and exceptional exit; do not swallow unexpected
exceptions merely to produce a normal exit. Tests must patch/restore SIGINT so
the existing loop's shutdown behavior does not contaminate subsequent tests.

The approved Windows empty-buffer Ctrl+Z alias preserves the conventional EOF
intent, but exits immediately rather than waiting for Enter. Unix suspend
behavior remains outside this scope.

## Testing strategy (after approval)

Use `create_pipe_input`, `DummyOutput`, and injected session dependencies to run
real key processing without a terminal or Ollama. Use bounded async execution
and explicit processing synchronization rather than sleep-based timing. The
[upstream testing guide](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/unit_testing.html)
describes these test I/O facilities. No async test dependency is necessary if
tests use `asyncio.run`.

Required deterministic cases:

1. Single-line Enter returns exactly one prompt; multiline composition submits
   one complete string, preserving text according to the approved whitespace rule.
2. Alt+Enter's Escape/Enter encoding and separately delivered fallback keys insert
   a newline at the cursor; neither accepts. Cover LF-as-Enter as well as CR.
3. Bracketed paste with LF, CRLF, trailing newline, and slash-looking lines stays
   in the buffer until explicit acceptance. Assert the prompt has not completed
   after processing paste, then submit. Include paste inside existing text.
4. Empty, spaces-only, and newlines-only Enter do not return or enter history;
   subsequent valid input still works.
5. Slash commands (including padded input) reach command handling unchanged in
   meaning and never call inference; multiline slash-prefixed text remains one
   command according to the existing matching rule.
6. Multiple reads share full multiline history; navigation boundaries, oldest/
   newest edges, draft restoration, recalled-text edits, duplicate policy, and
   `/clear` independence work as specified.
7. Up/Down within lines preserve the desired column and do not unexpectedly
   replace the buffer; cover short lines and the approved wrapping policy.
8. Prompt Ctrl+C with empty/nonempty text, Ctrl+D empty/nonempty behavior, actual
   EOF with a draft, Windows EOF alias, and fallback-path EOF are deliberate.
9. Spy on input raw-mode context exit and output reset/paste-disable/cursor
   restoration for interruption and EOF. A mocked `KeyboardInterrupt` alone is
   insufficient evidence of cleanup. Verify another prompt can run after unwind.
10. Loop tests mock the UI reader and inference, asserting exact multiline payload,
    no state changes for blank/interrupted input, completed turns, and unchanged
    rollback after generation interruption and engine failure. Use real `Session`
    snapshots for rollback assertions, not only a mocked rollback call.
11. Non-TTY fallback selection and line-oriented behavior remain usable without
    editor control sequences. Keep existing rendering/command/session tests.

Iteration: `python -m pytest tests/test_ui.py tests/test_cli.py` (or focused test
names). Before completion run and report exact commands/results:

```text
pytest
ruff check .
mypy src
python -m build
```

If executables require interpreter-qualified forms, record those exact commands.
Install build tooling into the development environment if needed; it is not a
runtime dependency. Existing CI is Ubuntu/Python 3.10 with tests, lint, and typing;
it does not establish Windows/macOS or packaging verification. Record baseline
failures separately and do not silently widen this feature into unrelated cleanup.

## Risks and approved decisions

The user approved the proposed choices with "Proceed" on 2026-09-23:

| Decision | Recommendation / tradeoff |
| --- | --- |
| Newline fallback | Escape then Enter; keep LF compatible with Enter rather than advertise Ctrl+J. |
| Whitespace | Preserve nonblank prompt text verbatim; deliberate departure from `.strip()`, valuable for pasted code. |
| History boundaries | First/last logical line; visual-row behavior is extra complexity. |
| Unmarked paste | Document the inability to guarantee safe paste without identifiable paste events; add explicit paste mode only if requested. |
| EOF keys | Ctrl+D empty exits everywhere; Windows Ctrl+Z empty is an immediate EOF alias. |
| Non-TTY behavior | Retain line-oriented fallback; no new headless prompt framing. |

Additional risks: terminal-reserved Alt shortcuts, Escape sequence timing,
platform-dependent native input events, default bindings that accidentally bypass
blank validation, and tests that assert configuration without exercising events.
In-memory history grows for the process lifetime and retains submitted text,
including failed prompts; no disk privacy footprint or new retention policy.
Large pastes can exceed model context; context budgeting remains out of scope.

## Ordered implementation steps

1. Review this plan, resolve decisions above, and record explicit user approval.
   Do not begin implementation before that approval.
2. Check current branch/worktree and follow `CONTRIBUTING.md` when starting the
   implementation branch; preserve unrelated work. Capture baseline checks.
3. Add/install the dependency and verify Python compatibility and relevant library
   behavior. Update this plan if observed behavior contradicts assumptions.
4. Implement the UI reader, acceptance/newline bindings, blank guard, memory
   history, EOF policy, cleanup boundary, and non-TTY selection with focused tests.
5. Reuse the reader in the application loop; adapt command detection only if the
   approved whitespace policy requires it. Keep inference/session lifecycle intact.
6. Add event-driven input tests and loop regression tests described above. Resolve
   library-default conflicts with small explicit bindings, not custom terminal code.
7. Update banner/README with approved keys, history lifetime, and limitations.
8. Run all required checks and available manual terminal smoke tests. Record exact
   outcomes and untested platforms here; report remaining limitations honestly.

## Acceptance criteria

- Enter submits one complete nonblank prompt; newline keys only insert newlines.
- Identifiable multiline paste preserves line structure and never submits partial
  lines, including trailing newlines; unsupported paste behavior is documented.
- Cursor and history navigation follow the approved boundary policy, preserve
  multiline entries and drafts, and retain history across reads in the process.
- Slash commands work through the same interface; blank input reaches neither
  commands nor inference. No persistent input history is introduced.
- Input is UI-owned; session commit/rollback and generation interruption behavior
  remain unchanged and are covered by deterministic tests.
- Ctrl+C/EOF follow the approved policy and terminal modes restore on exit.
- Windows, macOS, and Linux support relies on the chosen library; actual tested
  terminals and remaining limitations are recorded separately from expectations.
- Required tests, lint, typing, and packaging checks have recorded results; no
  completion claim substitutes mocked cleanup for actual lifecycle coverage.
- No autocomplete, syntax highlighting, model selection, or output redesign.

## Decision and verification log

- 2026-09-23: Initial repository investigation and upstream documentation/source
  review complete. `prompt_toolkit` absent from the active project environment.
  No dependency installation, production edits, or implementation tests performed.
  Awaiting plan review and explicit implementation approval.
- 2026-09-23: User approved the plan. Created `feat/multiline-input` from the
  available `origin/main` reference. That base predates the separate
  `ci/quality-checks` branch inspected during planning: CI and its existing lint/
  typing fixes are not on this feature branch. No unrelated changes imported.
- Baseline on the implementation branch: `python -m pytest -q` passed 49 tests
  (one existing cache permission warning); `python -m ruff check .` reported 18
  errors; `python -m mypy src` reported 3 errors in commands/session. Initial
  `python -m build` could not run because build tooling was absent.
- Installed `prompt_toolkit` 3.0.53 (requires Python >=3.10), `wcwidth` 0.9.0,
  and development build tooling. The sandbox package lookup failed; installation
  succeeded with approved network access. No lockfile added.
- Implementation uses `ui.InputReader` with `read() -> str | None`. Interactive
  sessions retain raw text, while the non-TTY path retains its original trimming.
  A validator also prevents alternative library acceptance bindings from submitting
  whitespace. No engine, session, or command-handler changes were needed.
- Event tests exposed Alt+Enter with an LF encoding: added `escape`, `c-j` as
  another newline binding, while plain LF still submits. Both are tested.
- Editor tests live in `tests/test_input.py` to keep stream-rendering tests small.
  They use processing barriers and explicitly await the library's asynchronous
  history initialization before sending navigation keys. The synchronous reader
  is also exercised through multiple real prompt sessions with bounded timeouts.
- Implementation completed with in-memory history only, no inference/state access
  from the editor, and unchanged generation rollback. README and banner describe
  the approved keys. The plan's ordered implementation steps are complete except
  for manual terminal smoke checks, which were unavailable in this tool session.

### Final verification

Environment: Windows, Python 3.12.10, pytest 9.1.1, prompt_toolkit 3.0.53.

| Exact command | Result |
| --- | --- |
| `python -m pytest tests/test_input.py tests/test_ui.py tests/test_cli.py -q` | 61 passed, 1 existing cache permission warning. |
| `python -m pytest` | 88 passed, 1 existing cache permission warning, 11.92 seconds. |
| `python -m ruff check .` | Failed with 14 pre-existing errors in scripts, commands, engine, session, and engine tests; baseline had 18 errors. |
| `python -m ruff check src/lclaude/ui.py src/lclaude/cli.py tests/test_input.py tests/test_ui.py tests/test_cli.py` | Passed; all changed Python files are clean. |
| `python -m mypy src` | Failed with the same 3 baseline errors: missing return annotation and list variance in session, Any return in commands. No new errors. |
| `python -m build` | Passed with approved escalation after sandbox temporary-directory access was denied; produced `dist/lclaude-0.1.0.tar.gz` and `dist/lclaude-0.1.0-py3-none-any.whl`. |
| `git diff --check` | Passed. |

Also inspected built wheel metadata: Python >=3.10 and
`Requires-Dist: prompt-toolkit<4,>=3.0.52` are present. No Ollama inference or real
interactive terminal was required for tests. Lifecycle tests execute the actual
prompt application with pipe input and spy on raw-context exit, paste disable,
cursor restoration, and attribute reset; they cannot prove native console modes.

Remaining limitations: Alt/Option shortcut delivery depends on terminal mappings;
unmarked paste cannot safely distinguish newlines from submission; history follows
logical lines rather than wrapped screen rows. Native Windows Terminal, macOS,
and Linux terminal smoke tests, resize/Unicode visual inspection, and Python 3.10
execution were not performed here. Library support is not reported as manual
platform verification. The existing `.pytest_cache` access warning did not affect
test outcomes. Baseline lint/type fixes remain outside this feature's scope.
