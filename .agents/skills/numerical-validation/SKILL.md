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

Keep the permanent test parametrized through the public API where possible.
Register rpy2 test files in `tests/conftest.py` and use the strict R marker
matching dependency availability.

For behavior shared with `feols` or `fepois`, use the fixest-first harness
to diagnose a case before adding it permanently:

```bash
pixi run -e py312-r compare-fixest scripts/reference/cases/<case>.toml
```

The harness does not replace the permanent pytest comparison. Read
`docs/developer/reference-harness.md` before adding a case or recording
reference output.
