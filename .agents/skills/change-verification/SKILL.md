---
name: change-verification
description: Selects, runs, and reports the checks a pyfixest change requires. Use after an implementation stabilizes and before handing off any code, test, documentation, CI, or metadata change, and whenever a handoff message or PR body needs a verification report.
---

# Verify a pyfixest change

Test policy lives in `docs/developer/testing.md`; this skill is the procedure
that applies its "Runtime tiers", "Selection matrix", and "Release contract"
sections. Read those sections rather than the whole file.

Input: the diff against the resolved base, established as described under
"Establish the base" in `docs/developer/git-and-pr-style.md`. Output: the
report below, written into the handoff message or PR body rather than a
separate generated artifact.

## Procedure

1. List every changed path against the resolved base and classify the change
   with the "Selection matrix". Unknown or cross-cutting paths take the PR
   baseline. For a change confined to the documentation or workflow-metadata
   rows, the row is the whole procedure.
2. For a refactor declared invariant, run the release contract first and on
   every iteration; it is the cheapest check that can falsify the whole
   change. A failure reclassifies the change as numerics, as "Release
   contract" describes.
3. While editing, run the targeted tests and the changed-file lint and type
   checks for the touched seam. Once the implementation stabilizes, run the
   selected broader baseline once.
4. Assign each required long check to a local run or to exact-head CI. Defer
   only under the conditions in "Runtime tiers", after the targeted checks pass,
   never a failing check or a targeted check needed to understand unresolved
   risk, and name the check, the reason for deferral, the destination, and the
   head SHA under test.

## Report

For every applicable check record:

- status: passed, failed, deferred, or not run;
- the exact command and elapsed time;
- for a deferred check, the reason, destination, and head SHA;
- for the release contract, the passed case count or the skip reason.

Do not claim implementation handoff while a required local check is
unreported or failing, and do not claim merge readiness until all required
merge evidence has passed on the exact head.
