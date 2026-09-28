---
name: pr-handoff
description: Prepare pyfixest commits and draft PRs after verification, before pushing or submitting, or when curating agent-owned history.
---

# Hand off work for review

Read "Branch names", "Stacks", "Commits", "Pull-request opening", and
[Approval and handoff](../../../docs/developer/git-and-pr-style.md#approval-and-handoff)
in `docs/developer/git-and-pr-style.md`. This skill applies those policies.

Input: a verified branch and its immediate parent, with the verification
report from the `change-verification` skill. Output: a draft PR or stack whose
body follows the documented style. Required long checks may still be running
in exact-head CI when the body says so.

## 1. Decide whether history needs curation

Inspect `git log <parent>..HEAD` against "Commits". Keep coherent commits that
pair behavior with tests. Curate WIP, fixup, accidental, and unrelated
formatting commits when present; no fixed commit sequence is required.
If the log already meets the policy, skip to step 4. Otherwise prepare the
specific rewrite for approval.

## 2. Authorize the rewrite

Prepare the branch inventory and commands required by "Approval and handoff".
If that specific rewrite is not already approved in the current conversation,
ask for approval and stop before rewriting. Proceed only when all policy
preconditions are met.

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
verification summary. Submit as a draft and stop at handoff. Apply the
readiness and human-review requirements in "Approval and handoff"; do not merge.
