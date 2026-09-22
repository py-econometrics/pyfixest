---
name: implementation-strategy
description: Plan estimator, API behavior, inference, formula, post-estimation, and shared-core changes in pyfixest. Includes invariant refactors; excludes prose-only edits.
---

# Choose the implementation strategy

Input: the requested change. Output: a short plan of material decisions,
updated only when the scope or design changes. Apply the
[change-scope policy](../../../docs/developer/git-and-pr-style.md#change-scope).

## Placement and precedent

Find the nearest implementation and its tests. Choose the primary placement:
estimator add-on, post-estimation module, shared primitive, backend kernel, or
shared-core change. Use the
[extension seams](../../../docs/developer/architecture.md#repository-map-and-extension-seams)
and read the applicable architecture sections:

- New estimators: "Estimator add-ons" and "Result and numerical boundaries".
- Shared-core changes: "Stable core", including its design-approval boundary.
- Estimator-state changes: "Formula-state and lifecycle boundaries" and
  "Implemented array and weight domains".
- New or changed public entries: "Public documentation".

## Decisions before implementation

Record only applicable decisions; link existing contracts that remain unchanged.

- Target module, precedent, reused primitives, and public API/result impact.
- For estimator or inference behavior: supported weights, FE, IV, multiple
  estimation, retained-data modes, and backends, plus explicit unsupported
  errors, under "Result and numerical boundaries". New estimators need a
  complete support matrix; existing features need the affected paths reviewed.
- For new estimators or numerical changes: the permanent external reference
  and test location, selected through
  [External numerical references](../../../docs/developer/testing.md#external-numerical-references).
  Resolve missing support or reference decisions before implementing that path.
- For changes that can affect numerical results: whether results must remain
  invariant or which quantities and estimators may change. Apply
  [Release contract](../../../docs/developer/testing.md#release-contract);
  intentional differences from its baseline require explicit reasons.
- Applicable exports, documentation, compatibility-ledger, and changelog work.

Carry only decisions that matter to reviewers into the PR; do not reproduce
this checklist or an unchanged support matrix.
