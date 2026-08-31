---
name: implementation-strategy
description: Classify and place pyfixest estimator, inference, formula, API, and core changes before implementation.
---

# Choose the implementation strategy

Use this skill before changing a public estimation surface or shared estimation
code. Read `AGENTS.md` and the nearest implementation and tests first.

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

Do not put estimator-specific switches into the generic runner or numerical
logic into model classes. Stop for maintainer design approval before expanding
core merely to accommodate one new estimator.

## Produce the strategy

Before editing, record:

1. the primary classification and target module;
2. the nearest in-repo precedent;
3. the proposed public names and signatures, their consistency with sibling
   APIs, and result-object or `FixestMulti` exposure;
4. reused primitives and any proposed shared primitive;
5. behavior for `aweights`, `fweights`, fixed effects, IV, multiple
   estimation, `lean`, `store_data`, and relevant backends;
6. explicit unsupported paths and their errors;
7. the external numerical reference and permanent-test location;
8. documentation, exports, and changelog wiring.

For post-estimation work, keep fitted-model methods to validation, state
extraction, and delegation. If a method would also own numerical iteration or
summarization, move that work to the standalone post-estimation module. For
resampling methods, decide before implementation whether failed replicates are
replaced, retained as missing, or fatal; do not let the loop structure choose
that public contract implicitly.

If the support matrix or external reference is unresolved, resolve it before
implementation rather than allowing the code to choose policy implicitly.
