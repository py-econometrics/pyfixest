---
name: history-curation
description: Safely curate agent-owned pyfixest commits before first PR submission, including layers managed by gh stack.
---

# Curate commit history

Use this skill after code and checks stabilize, before the first remote PR
submission. Read Phase 5 of `.agents/feature-pr.md`.

## Authorization and preconditions

Proceed only when:

- the worktree is clean;
- the branch is named, is not `master`, and its immediate parent is verified;
- the history is agent-owned and unpushed, or the user explicitly approved a
  rewrite;
- the original tip SHA is recorded.

Never rewrite contributor-owned history. Never silently rewrite after review
starts.

## Curate one layer

Inspect `git log <parent>..HEAD`. Keep a few commits that tell the review
story: contracts/helpers/tests, implementation/wiring/tests, then exports/docs.
Remove WIP, fixup, accidental, and formatting-only commits. Pair tests with the
behavior they establish and run the applicable targeted check for every
reconstructed commit.

Use interactive rebase when practical. Otherwise use the guarded soft-reset
procedure in the feature workflow. Confirm the final tree is identical to the
recorded tip with `git diff --exit-code <tip> HEAD`.

For a stack, curate bottom-up and rebase descendants with
`gh stack rebase --upstack`. After remote submission, use `gh stack push`
instead of raw force push and obtain maintainer approval before cleanup that
would invalidate reviews.
