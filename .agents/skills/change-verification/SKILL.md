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

For a refactor declared invariant, follow the release-contract cadence in
`docs/developer/testing.md`. Run focused behavior and caller tests while a
related change batch is active, then satisfy the contract requirement at
stabilization using still-applicable evidence. Run missing or invalidated checks,
normally once on the cumulative head. Run the contract earlier when focused
tests leave shared numerical risk unresolved or it is needed for investigation
or baseline setup. Cite it only when it reports passed cases: a skip means no
baseline, not success. A failure ends the refactor classification. Fix it or
follow the numerics row; never widen a contract tolerance to get green.

Start with targeted tests and changed-file format/lint/type checks. Once the
implementation stabilizes, identify and satisfy still-missing required evidence
using the policy cadence, and assign each required long check to a local run or
exact-head CI.

Unknown paths require the conservative PR baseline. Apply the general cadence
to initial work, batches of review feedback, and stack reconstruction. Run
focused checks for each independently functional layer against its immediate
parent, and coordinate one owner to run broad suites once on the stabilized
cumulative head against the trunk where a new run is needed. Repeat them only
when later work invalidates the evidence, unresolved risk requires it, or an
explicit acceptance requirement does. Run broad suites per layer only when each
layer must be independently releasable.

Complete the required edit and handoff checks locally. A long check listed as
merge evidence may be deferred to exact-head CI when a local run would add no
diagnostic value. Defer only after the targeted checks pass, and identify the
check, destination, and head under test. Never defer a failing check or a
targeted check needed to understand unresolved risk.

## Report truthfully

For every applicable check record:

- status: passed, failed, deferred, or not run;
- exact command;
- revision, environment, and tested scope;
- elapsed time;
- reason and destination for a deferred check;
- for the release contract, the passed case count or the skip reason.

Evidence from an earlier revision may still cover an unchanged seam, but must
not be reported as exact-head evidence. State the intervening scope and rerun
only what it invalidated.

Do not claim implementation handoff while required local checks are unreported
or failing, and do not claim merge readiness until all required exact-head
evidence has passed.

Write this report directly in the handoff or PR body. Do not introduce a
generated verification artifact unless the task specifically requires one.
