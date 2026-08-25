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

Read the error and tolerance contract in `docs/developer/testing.md`. Name the
quantity in every numerical assertion, use separate tolerances for coefficients,
inference, residuals, and predictions, and compare discrete structure exactly.
Use `tests/test_vs_fixest.py` as the current precedent for
`feols`/`fepois`/`feglm` and `tests/test_quantreg.py` for `quantreg`; do not copy
the loosest tolerance from either file into unrelated cases.

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

Extend the existing permanent public-API matrix when a new case fits it. The
fast matrix provides focused feedback but does not replace estimator-specific
coverage needed for a change.
