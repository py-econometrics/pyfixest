---
name: history-curation
description: Safely curate agent-owned pyfixest commits before first PR submission, including layers managed by gh stack.
---

# Curate commit history

Use this skill after code and checks stabilize, before the first remote PR
submission. Read the branch and commit conventions in
`docs/developer/git-and-pr-style.md`.

## Authorization and preconditions

Before any rewrite, report:

- every branch and its exact immediate parent;
- whether each branch is agent-owned, pushed, or under review;
- the original tip SHA for every affected branch;
- dependent branches that will need rebasing; and
- the exact rewrite and stack-rebase commands.

Then ask the user explicitly whether that specific rewrite is approved. General
permission to implement, commit, or open a PR does not authorize rewriting
history. Proceed only when the user approves in the current conversation and:

- the worktree is clean;
- the branch is named, is not `master`, and its immediate parent is verified;
- the history is agent-owned;
- the original tip SHA is recorded.

Never rewrite contributor-owned history. Never silently rewrite after review
starts.

## Curate one layer

Inspect `git log <parent>..HEAD`. Keep a few concise commits that tell the
review story: contracts/helpers/tests, implementation/wiring/tests, then
exports/docs. Each commit must be self-contained, address one reviewer concern,
and be small enough for a human to review independently. Pair tests with the
behavior they establish and require every commit to pass its applicable
targeted checks. Split a commit when a reviewer would need to understand an
unrelated concern or an unnecessarily long diff at the same time.

Remove WIP, fixup, accidental, and formatting-only commits.

Use interactive rebase when practical. Otherwise, after recording the tip and
verifying the layer base, use `git reset --soft <verified-layer-base>` and
reconstruct one approved slice at a time. Confirm the final tree is identical
to the recorded tip with `git diff --exit-code <tip> HEAD`, then inspect the
resulting log and diff.

For a stack, curate bottom-up and rebase descendants with
`gh stack rebase --upstack`. After remote submission, use `gh stack push`
instead of raw force push and obtain maintainer approval before cleanup that
would invalidate reviews.
