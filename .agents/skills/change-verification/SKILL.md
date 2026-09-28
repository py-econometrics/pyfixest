---
name: change-verification
description: Select, run, and report required checks for a pyfixest change before handoff or PR submission.
---

# Verify a pyfixest change

Input: the diff against the
[resolved base](../../../docs/developer/git-and-pr-style.md#establish-the-base),
including uncommitted and untracked changes. Output: verification evidence
summarized in the handoff or PR.

## Select and run checks

1. Classify every changed path with the
   [Selection matrix](../../../docs/developer/testing.md#selection-matrix).
   For documentation or workflow-metadata-only changes, follow those rows;
   numerical suites and release-contract guidance do not apply.
2. For code changes, read "Runtime tiers" and the commands needed for the
   selected checks. Run targeted tests and changed-file lint/type checks while
   editing, then the selected broader baseline once the implementation settles.
3. For numerical changes or invariant refactors, read "Release contract".
   Run it early for an invariant refactor and rerun after edits that can affect
   results. Reuse a passing result after unrelated prose or metadata edits.
   Investigate failures under that policy; do not relabel drift as intentional
   merely to make the check pass.
4. Reuse applicable evidence for the same code state. Defer required long
   checks only under "Runtime tiers"; never defer failing checks or targeted
   checks needed to resolve a material uncertainty.

Report using
[Verification reporting](../../../docs/developer/testing.md#verification-reporting).
Required local checks must pass and be reported before implementation handoff;
all required merge evidence must pass on the exact head before claiming merge
readiness.
