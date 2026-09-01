---
name: numerical-validation
description: Design and verify external numerical references for pyfixest estimators, inference, solvers, formulas, weights, and kernels.
---

# Establish numerical correctness

Use this skill for new estimators and any change that can alter numerical
results. Read the numerical-reference policy in `AGENTS.md`.

## Hard gates

- Every new estimator needs a permanent comparison with existing software.
- Numerical changes to existing estimators need an external comparison wherever
  overlapping software exists.
- Shape tests, internal reimplementations, closed-form cases, and simulations
  are supplemental; they do not replace the external comparison for a new
  estimator.
- If no existing implementation is available, the new estimator is not
  merge-ready.

## Choose the reference

Prefer, in order:

1. live R `fixest`, another established R package, or a well-established Python
   package available in a maintained environment;
2. a CRAN-only R package in the extended environment;
3. stored output from Stata or other established software, with the generating
   script and version committed;
4. another established external package with its exact version recorded.

Use the same deterministic rows and model specification on both sides. Compare
named values so ordering differences cannot hide discrepancies.

## Required evidence

Record the software/version, formula, data or seed, weights, vcov/SSC, supported
paths, outputs, and explicit `rtol`/`atol` with justification. Compare the
quantities the method promises: coefficients, vcov, standard errors, degrees of
freedom, observations, dropped terms, convergence, or deterministic prediction
subsets as applicable.

Read the error, tolerance, and suite-growth contracts in
`docs/developer/testing.md`. Name the compared quantity, use its canonical
estimator-specific tolerance, and extend the nearest permanent matrix before
adding a new test function or file.

Keep the permanent test parametrized through the public API where possible and
keep its runtime suitable for regular CI use. Use the release numerical-contract
snapshots for broad repeatable regression feedback, but never treat a released
pyfixest result as an external correctness reference. Live R comparisons must run
through rpy2 inside pytest. Register every rpy2-importing test file in
`tests/conftest.py` and use the strict R marker matching dependency
availability. Do not replace an available live reference merely to shorten the
edit loop; first select affected live cases or use the fast matrix. Follow the
stored-reference criteria in `docs/developer/testing.md` when live execution is
unavailable, unreliable, or impractical.

Release snapshots are generated in a platform-local gitignored cache from the
pinned release environment, never from development head. The snapshot task
prepares and invalidates that cache automatically. When a snapshot drifts,
inspect the post-baseline changelog first. Narrowly comment a comparison or case
only when it is a documented intentional behavior change, and retain external R
evidence; leave unexplained drift as an ordinary failure for human review.
Change the pinned version only for an explicit baseline roll.

Use R `fixest` as the default reference for `feols`, `fepois`, and `feglm`.
Use R `quantreg` for `quantreg`. The fast direct matrix is available as:

```bash
pixi run -e py312-r test-r-fixest-fast
```

Use targeted Python or live-R cases while editing. Run the release snapshot
matrix once after stabilization, directly or through `test-py`, then produce
the affected canonical evidence locally or in exact-head CI. The fast live-R
suite does not replace permanent estimator-specific coverage or complete merge
evidence.
