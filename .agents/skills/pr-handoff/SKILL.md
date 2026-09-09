---
name: pr-handoff
description: Curates agent-owned pyfixest commits and submits a reviewer-ready draft PR or stack. Use after implementation and required verification stabilize, before the first push or PR creation, and whenever agent-owned commit history needs rewriting.
---

# Hand off work for review

The conventions live in `docs/developer/git-and-pr-style.md` under "Branch
names", "Stacks", "Commits", and "Pull-request opening"; this skill is the
procedure and its safety gates.

Input: a verified branch and its immediate parent, with the verification
report from the `change-verification` skill. Output: a draft PR or stack whose
body follows the documented style. Required long checks may still be running
in exact-head CI when the body says so.

## 1. Decide whether history needs curation

Inspect `git log <parent>..HEAD` against "Commits". The target is
a few commits that tell the review story: contracts/helpers/tests,
implementation/wiring/tests, then exports/docs. WIP, fixup, accidental, and
formatting-only commits must go. If the log already meets that bar, skip to
step 4. Otherwise curation is a history rewrite and needs step 2.

## 2. Authorize the rewrite

A rewrite is the one irreversible step in this procedure, so it is gated even
when the user has already asked for a PR. Before any rewrite, report:

- every branch and its exact immediate parent;
- whether each branch is agent-owned, pushed, or under review;
- the original tip SHA for every affected branch;
- dependent branches that will need rebasing;
- the exact rewrite and stack-rebase commands.

Then ask whether that specific rewrite is approved and end the turn. Proceed
only with approval given in the current conversation and when the worktree is
clean, the branch is named and is not `master`, its parent is verified, the
history is agent-owned, and the tip SHA is recorded. Never rewrite
contributor-owned history, and never rewrite silently after review starts.

## 3. Rewrite one layer

Use interactive rebase when practical. Otherwise, after recording the tip and
verifying the layer base, run `git reset --soft <verified-layer-base>` and
reconstruct one approved slice at a time. Confirm the final tree matches the
recorded tip with `git diff --exit-code <tip> HEAD`, then inspect the log and
diff. For a stack, curate bottom-up and rebase descendants with
`gh stack rebase --upstack`; after submission use `gh stack push` rather than
a raw force push.

## 4. Submit and stop

Choose one PR or a stack by "Stacks", inspect every layer's diff and the
cumulative diff, and write the body per "Pull-request opening" with the
verification report included. Submit as a draft.
Mark a layer ready for review only when its required checks pass or its long
checks are visibly running in exact-head CI, and do not call it merge-ready
until those pass.

Stop at handoff. Do not merge, and do not invoke `gh stack merge`.
