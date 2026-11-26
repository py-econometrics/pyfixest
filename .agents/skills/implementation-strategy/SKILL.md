---
name: implementation-strategy
description: Choose the implementation and numerical evidence for pyfixest estimator, inference, formula, post-estimation, public API, or shared-core changes. Includes invariant refactors; excludes explanations and prose-only edits.
---

# Implement a pyfixest change

Find the nearest implementation, its callers, and its tests, and apply the
[scope policy](../../../docs/developer/git-and-pr-style.md#change-scope).
Then settle three things before writing code:

- **Support matrix.** Which paths the change touches under
  [results and numerical boundaries](../../../CODING_STANDARDS.md#results-and-numerical-boundaries).
  A new estimator needs the complete matrix; an existing feature needs only
  the affected paths. Reject unsupported combinations explicitly.
- **Independent reference.** For numerical work, a reference chosen under
  [reference selection](../../../docs/developer/testing.md#choosing-a-numerical-reference).
- **Invariance.** Whether results must stay unchanged. Read "Release
  contract" for the baseline check; do not silently turn a refactor into a
  behavior fix.

For a small fix, these can be a brief note in the ongoing work. Write a
separate plan only when it helps coordinate substantial work or the user asks
for one. Verify through `change-verification` and review the final diff
through `pr-review`, without restarting the workflow at handoff.
