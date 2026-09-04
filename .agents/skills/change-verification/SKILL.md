---
name: change-verification
description: Select, run, and report pyfixest checks before handing off code, tests, documentation, CI, or metadata changes.
---

# Verify a pyfixest change

Use this skill after implementation stabilizes and before handoff.

## Select checks by risk

Resolve and record the actual PR base and merge base as described in
`docs/developer/git-and-pr-style.md`; do not assume local `master` is current.
Inspect every changed path against that base, then classify the change and pick
its checks with the selection matrix in `docs/developer/testing.md`. That matrix
is authoritative, including for which changes justify a documentation build.
For a change confined to the documentation or workflow-metadata rows of that
matrix, the row is the whole procedure: run its listed checks and report them.

For a refactor declared invariant, run the release contract first and on every
iteration; it is the cheapest check that can falsify the whole change. Cite it
only when it reports passed cases: a skip means no baseline, not success. A
failure ends the refactor classification. Fix it or follow the numerics row.
Never widen a contract tolerance to get green.

Start with targeted tests, changed-file format and lint checks, and the
whole-package type check. Once the implementation stabilizes, run the selected
broader baseline once and assign each required long check to a local run or
exact-head CI.

Unknown paths require the conservative PR baseline. For a stack, run targeted
checks for each layer against its immediate parent and the broad suites once on
the cumulative top against the trunk, unless each layer must be independently
releasable.

Complete the required edit and handoff checks locally. A long check listed as
merge evidence may be deferred to exact-head CI when a local run would add no
diagnostic value. Defer only after the targeted checks pass, and identify the
check, destination, and head under test. Never defer a failing check or a
targeted check needed to understand unresolved risk.

## Report truthfully

For every applicable check record:

- status: passed, failed, deferred, or not run;
- exact command;
- elapsed time;
- reason and destination for a deferred check;
- for the release contract, the passed case count or the skip reason.

Do not claim implementation handoff while required local checks are unreported
or failing, and do not claim merge readiness until all required exact-head
evidence has passed.

Write this report directly in the handoff or PR body. Do not introduce a
generated verification artifact unless the task specifically requires one.
