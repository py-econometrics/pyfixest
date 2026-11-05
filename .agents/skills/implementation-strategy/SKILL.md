---
name: implementation-strategy
description: Choose the implementation and numerical evidence for pyfixest estimator, inference, formula, post-estimation, public API, or shared-core changes. Includes invariant refactors; excludes explanations and prose-only edits.
---

# Implement a pyfixest change

Find the nearest implementation, its callers, and its tests. Use the
[architecture seams](../../../docs/developer/architecture.md#repository-map-and-extension-seams)
to choose where the change belongs. Apply the
[scope policy](../../../docs/developer/git-and-pr-style.md#change-scope).

Resolve the decisions that affect this change:

- The user-visible outcome or invariant, and the simplest implementation that
  preserves the relevant contracts. Read "Stable core" for contract changes;
  an already-authorized design does not need another approval.
- Affected support paths under
  [result boundaries](../../../docs/developer/architecture.md#result-and-numerical-boundaries).
  New estimators need a complete support matrix; existing features need only
  the affected paths reconsidered. Reject unsupported combinations explicitly.
- For numerical work, an independent reference under
  [reference selection](../../../docs/developer/testing.md#choosing-a-numerical-reference),
  and whether results must stay invariant. Read "Release contract" for the
  applicable baseline check; do not silently turn a refactor into a behavior fix.
- Necessary exports, user documentation, and compatibility decisions.

For a small fix, these decisions can be a brief explanation in the ongoing
work. Write a separate plan only when it helps coordinate substantial work or
the user asks for one. Implement once the material choices are clear; reuse
settled decisions. Verify through `change-verification` and review the final
diff through `pr-review`, without restarting the workflow at handoff.
