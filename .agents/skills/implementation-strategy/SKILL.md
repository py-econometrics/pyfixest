---
name: implementation-strategy
description: Classify and place pyfixest estimator, inference, formula, API, and core changes before implementation.
---

# Choose the implementation strategy

Use this skill before changing a public estimation surface or shared estimation
code. Read `docs/developer/architecture.md` and the nearest implementation and
tests first.

## Classify the change

Choose exactly one primary placement:

- **Estimator add-on:** standalone API or domain module composing stable
  primitives. This is the default for a new estimator.
- **Post-estimation:** standalone numerical module with thin fitted-model
  wrappers.
- **Shared primitive:** generic internal operation with a real shared consumer.
- **Backend kernel:** measured, non-vectorizable hot loop with a readable
  reference implementation where feasible.
- **Core change:** modification to formula planning, model matrices, generic fit
  orchestration, inference contracts, or result interfaces.

Stop for maintainer design approval before expanding core merely to accommodate
one new estimator.

## Produce the strategy

Before editing, record the following in the plan you present to the user, and
carry the durable parts into the PR body's opening paragraph:

1. the primary classification and target module;
2. the nearest in-repo precedent;
3. public API and result-object impact;
4. reused primitives and any proposed shared primitive;
5. behavior for `aweights`, `fweights`, fixed effects, IV, multiple
   estimation, `lean`, `store_data`, and relevant backends;
6. explicit unsupported paths and their errors;
7. the external numerical reference and permanent-test location, chosen by the
   preference order in `docs/developer/testing.md`;
8. documentation, exports, and changelog wiring;
9. whether results are intended to be invariant. If yes, route release-contract
   timing through the risk-based cadence in `docs/developer/testing.md`; any
   failure is a regression. If no, list the quantities and estimators expected
   to move, because each becomes a `reason`ed declaration in
   `tests/test_release_contract.py`.

If the support matrix or external reference is unresolved, resolve it before
implementation rather than allowing the code to choose policy implicitly.
