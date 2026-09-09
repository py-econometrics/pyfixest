---
name: implementation-strategy
description: Classifies and places a pyfixest change before implementation and records its support matrix, external reference, and invariance intent. Use before any estimator, public API, inference, formula, post-estimation, or shared-core work, including refactors that must not change results.
---

# Choose the implementation strategy

Architecture policy lives in `docs/developer/architecture.md` and test policy
in `docs/developer/testing.md`; this skill is the procedure that applies them
before code exists, so that the code never chooses policy implicitly. Read
"Stable core", "Estimator add-ons", and "Result and numerical boundaries" in
`architecture.md`, "External numerical references" in `testing.md`, and the
nearest in-repo implementation and its tests first. Read the estimator-state
sections of `architecture.md` only for shared-core or estimator-state work.

Input: the task. Output: the strategy below, presented in the plan and carried,
in its durable parts, into the PR body's opening paragraph.

## Classify the change

Choose exactly one primary placement:

- **Estimator add-on:** standalone API or domain module composing stable
  primitives. This is the default for a new estimator.
- **Post-estimation:** standalone numerical module with thin fitted-model
  wrappers.
- **Shared primitive:** generic internal operation with a current shared
  consumer.
- **Backend kernel:** measured, non-vectorizable hot loop with a readable
  reference implementation where feasible.
- **Core change:** modification to formula planning, model matrices, generic
  fit orchestration, inference contracts, or result interfaces.

Expanding the stable core to accommodate one estimator needs maintainer design
approval before implementation.

## Record the strategy

1. the primary classification and target module;
2. the nearest in-repo precedent;
3. public API and result-object impact;
4. reused primitives and any proposed shared primitive;
5. behavior for `aweights`, `fweights`, fixed effects, IV, multiple
   estimation, `lean`, `store_data`, and relevant backends;
6. explicit unsupported paths and their errors;
7. the external numerical reference and permanent-test location, chosen by
   the preference order under "External numerical references" in
   `testing.md`;
8. documentation, exports, and changelog wiring;
9. whether results are intended to be invariant. If yes, the release contract
   is the edit-loop gate and any failure is a regression (see "Release
   contract" in `testing.md`). If no, list the
   quantities and estimators expected to move; each becomes a `reason`ed
   declaration in `tests/test_release_contract.py`.

Resolve an unsettled support matrix or external reference before
implementation rather than while writing the code.
