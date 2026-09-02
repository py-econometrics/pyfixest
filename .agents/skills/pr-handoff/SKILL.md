---
name: pr-handoff
description: Curate agent-owned pyfixest commits and prepare a reviewer-ready single or stacked draft PR.
---

# Hand off work for review

Use this skill after implementation and required verification stabilize, before
the first remote submission. Required long checks may still be running in
exact-head CI when clearly reported. Branch, commit, and PR-body conventions
live in `docs/developer/git-and-pr-style.md`.

## Decide whether history needs curation

Inspect `git log <parent>..HEAD`. The target is a few concise commits that tell
the review story: contracts/helpers/tests, implementation/wiring/tests, then
exports/docs. Each commit must be self-contained, address one reviewer concern,
be small enough for a human to review independently, pair tests with the
behavior they establish, and pass its applicable targeted checks. WIP, fixup,
accidental, and formatting-only commits must go.

If the log already meets that bar, skip to "Choose the delivery shape".
Otherwise curation is a history rewrite and needs the authorization below.

## Authorize the rewrite

Before any rewrite, report:

- every branch and its exact immediate parent;
- whether each branch is agent-owned, pushed, or under review;
- the original tip SHA for every affected branch;
- dependent branches that will need rebasing; and
- the exact rewrite and stack-rebase commands.

Then ask the user explicitly whether that specific rewrite is approved. General
permission to implement, commit, or open a PR does not authorize rewriting
history. Proceed only when the user approves in the current conversation and the
worktree is clean, the branch is named and is not `master`, its immediate parent
is verified, the history is agent-owned, and the original tip SHA is recorded.

Never rewrite contributor-owned history. Never silently rewrite after review
starts.

## Rewrite one layer

Use interactive rebase when practical. Otherwise, after recording the tip and
verifying the layer base, use `git reset --soft <verified-layer-base>` and
reconstruct one approved slice at a time. Confirm the final tree is identical
to the recorded tip with `git diff --exit-code <tip> HEAD`, then inspect the
resulting log and diff.

For a stack, curate bottom-up and rebase descendants with
`gh stack rebase --upstack`. After remote submission, use `gh stack push`
instead of raw force push.

## Choose the delivery shape

Use one PR for a small cohesive change. Prefer a stack when two or more
independently reviewable concerns form coherent dependency layers. Each layer
must be testable against its immediate parent and contain no unrelated cleanup.

Before submission, inspect every layer's log/diff and the cumulative diff.
Identify the immediate parent only when it is not obvious from the PR base.

## Draft the body and stop

Write the body following `docs/developer/git-and-pr-style.md`. For numerical or
estimator changes, add only the material facts: `fixest` parity or the
intentional deviation and support limits; external reference and tolerance
rationale; failed or deferred verification and any performance impact. Link to
tests or policy instead of copying the support matrix into the body. Do not
force estimator-specific boilerplate into documentation, CI, or maintenance PRs.

Submit new PRs as drafts. Mark a layer ready for review only when required
checks pass or clearly identified long checks are visibly running in exact-head
CI. Do not describe the change as merge-ready until every required exact-head
check passes.

Stop at handoff. Do not merge the PR or invoke `gh stack merge`.
