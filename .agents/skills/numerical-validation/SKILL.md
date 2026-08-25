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

1. live R `fixest` or another established package available on conda-forge;
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

Keep the permanent test parametrized through the public API where possible.
Live R comparisons must run through rpy2 inside pytest. Register every
rpy2-importing test file in `tests/conftest.py` and use the strict R marker
matching dependency availability. Stored output is an exception only when the
external software cannot be run reliably in the test environment; commit its
generator and version alongside the values.

Use R `fixest` as the default reference for `feols`, `fepois`, and `feglm`.
Use R `quantreg` for `quantreg`. The fast direct matrix is available as:

```bash
pixi run -e py312-r test-r-fixest-fast
```

The fast suite provides focused edit feedback. It does not replace or establish
permanent estimator-specific coverage.
