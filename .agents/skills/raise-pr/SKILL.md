---
name: raise-pr
description: Prepare, open, or update a GitHub pull request for lclaude using its saved conventions and templates. Use when the user requests PR preparation or publication for repository changes.
---

# Raise a pull request

Work from the repository root. Read `AGENTS.md`, `CONTRIBUTING.md`, and
`.github/pull_request_template.md`; these are the canonical rules and PR format.
Use Git for local history and `gh` for GitHub operations. Use `gh api` for data
not conveniently exposed by ordinary `gh` commands; paginate list requests.

## Establish scope

Distinguish local preparation from publication using the user's request and
existing session authorization. Opening a PR includes the necessary scoped commit
and push; it does not include merging. Preparing alone stops at a local result
and proposed PR title/body. Do not add approval steps for already authorized work.

Inspect `git status`, staged and unstaged diffs, current branch, remotes, and recent
commits. Resolve the destination repository and default branch from actual remote
metadata, not a hardcoded owner. Check `gh` authentication when GitHub access is
needed. Inspect the associated issue when one is provided.

Preserve unrelated edits and staged work. If ownership of changes is ambiguous,
prepare what is clear and ask about the ambiguous portion before including it.
Use an existing task branch when appropriate; otherwise create a conventionally
named branch without losing local changes. Inspect the full diff and commits
against the intended base, including unexpected commits inherited by the branch.

## Prepare a reviewable result

Run checks relevant to the change using `AGENTS.md`. Fix failures introduced by
the task; report unrelated or environmental failures without expanding scope.
Review the final diff for accidental files and scope drift. Stage intended paths
or hunks and inspect the staged diff before committing when publication or a
commit is authorized. Do not include unrelated staged files in that commit.

Write the PR title using the repository convention. Fill the saved template with
the concrete problem, final behavior, actual verification results, and material
limitations. Remove unused optional sections and template comments. Use closing
issue keywords only for full resolution. Keep a local-only draft outside tracked
source files unless the user requests a saved artifact.

## Publish or update

Before a mutation, verify that it is within the user's authorized scope. Push the
intended branch to the verified remote. If a push is rejected, inspect divergence;
do not force-push as an automatic retry.

Find an existing open PR using the destination repository, head owner/branch, and
base branch. Update that PR when it matches the task. If only a closed or merged
PR exists, inspect which commits remain before deciding whether a new PR is needed.
Use explicit repository, base, and head arguments when creating a PR, and always
pass `--draft`. Completed work also starts as a draft. Mark ready only on an
explicit user request. Preserve an existing PR's review state when updating it
unless the user asks to change that state.

Follow the single-commit preference in `CONTRIBUTING.md`. When an authorized
update folds changes into a published commit, verify the remote head and push with
an explicit `--force-with-lease=<ref>:<expected-sha>`. Stop if the lease fails;
inspect new remote work before proceeding. Update the commit message to describe
the complete final scope. A future authorized merge should squash into `main`.

Pass multiline descriptions through a UTF-8 temporary file and `--body-file`.
Keep the file outside the tracked source tree. Do not embed arbitrary prose in
shell command strings. If creation or update times out, query GitHub before
retrying to avoid duplicate mutations. If state cannot be verified, report the
uncertainty and stop retrying mutations.

Inspect PR checks for the latest pushed commit. Distinguish passed, failed,
pending, and absent checks. If checks are still pending, report that accurately;
monitor to completion when the user requests it. Do not merge, enable auto-merge,
or bypass repository rules as part of opening a PR.

Return the PR link (or local title/body for preparation), a concise change summary,
verification outcomes, and remaining blockers. Do not describe a PR as merged or
CI as passed without confirming that state.
