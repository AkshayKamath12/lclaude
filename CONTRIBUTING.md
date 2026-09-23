# Contributing to lclaude

Read [AGENTS.md](AGENTS.md) for architecture and verification commands, and
[docs/hld.md](docs/hld.md) for planned direction. Keep each change focused on one
reviewable outcome.

## Branches and commits

Use a short branch named `<type>/<description>`, optionally including the issue
number: `feat/8-multiline-input`, `fix/interrupted-stream`, or
`refactor/session-lifecycle`. Use `docs/`, `test/`, `ci/`, or `chore/` when appropriate.
Start new work from the current default branch (currently `main`); retain an
existing task branch when continuing its work.

Use commit and PR titles of the form `<type>: <description>`, with an optional
scope: `fix(session): roll back interrupted responses`. Describe the outcome in
the imperative. Use `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, or `chore`.
Choose the type for the primary change; do not relabel a behavior change as a refactor.

Inspect the working tree and staged diff before committing. Stage only files or
hunks belonging to the task. Preserve unrelated changes, including existing
staged work. Do not discard work or rewrite shared history as routine cleanup.

## Pull requests

Use [.github/pull_request_template.md](.github/pull_request_template.md). Lead with
the problem and resulting behavior, then explain only the implementation details
needed for review. Record the checks actually run and their outcomes. State why
relevant checks were skipped; documentation-only changes normally need document
and link validation rather than the Python test suite.

Use `Closes #123` only when the PR fully resolves that issue. For part of a larger
issue, use `Related to #123` and describe the remaining scope. Do not invent issue
links, reviewer assignments, or labels.

Always open PRs in draft mode, including completed and validated work. With `gh`,
pass `--draft` explicitly. Mark a PR ready for review only when the user explicitly
requests it. Explain any incomplete work or unresolved validation in the description.
Update an existing open PR for the same branch instead of creating a duplicate;
preserve its review state unless the user requests a change. Keep the description
aligned with the final diff.

Opening a PR does not imply permission to merge it. Follow the repository's current
review and merge rules. A missing CI run is not evidence that checks passed.

Prefer one commit per PR and one commit per merged change on `main`. Use squash
merge when merging is explicitly requested, with a commit title and body that
describe the complete final change. When folding follow-up edits into a published
commit, obtain authorization to rewrite that branch's history unless already given,
then use `--force-with-lease` against the verified remote commit. Never force-push
`main` to implement this convention. If repository rules prevent squash merging,
report the conflict rather than silently choosing another method.

## Issues

Use the bug or work-item templates in [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE).
Describe observable behavior and acceptance criteria. Break large phases into
independently verifiable work where practical. Include model, Ollama version, and
hardware details when relevant to inference failures; remove secrets and private
source code from logs.

## Agent workflow

The repository skill is [.agents/skills/raise-pr/SKILL.md](.agents/skills/raise-pr/SKILL.md).
Example requests:

- `Use $raise-pr to prepare a PR for these changes without publishing it.`
- `Use $raise-pr to open a PR for the completed work on issue #8.`
- `Use $raise-pr to update this branch's PR and check its CI status.`

These files define conventions; GitHub Actions and repository rules must be
configured separately to enforce checks and merge requirements.
