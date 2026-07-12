---
name: pyfixest
description: "Use for fixest-style econometrics in Python with PyFixest: OLS/WLS/IV, high-dimensional fixed effects, GLMs, quantile regression, inference, reporting, DiD, demeaning backends, or PyFixest errors. Read this skill and the bundled version-matched docs before guessing formula syntax, covariance options, or result APIs."
---

# PyFixest

Use the installed, version-matched documentation before relying on memory. Start
with pyfixest/docs/llms.txt or pyfixest/docs/index.md; use public functions
such as pyfixest.feols instead of internal modules.

## Workflow

1. Identify the estimator and whether the request needs fixed effects, IV,
   clustering/HAC, multiple estimation, or a specialized estimator.
2. Read the matching reference below, then confirm details in the installed docs
   page it names. Do not invent unsupported combinations.
3. Write a minimal public-API example. Use a seeded np.random.default_rng for
   synthetic data and retain the fitted result for post-estimation.
4. For an error, quote its received value and follow its local documentation
   pointer before changing the model specification.

## References

- [Core APIs](references/core-api.md) — estimator choice, result objects, and
  multiple estimation.
- [Formula syntax](references/formula-syntax.md) — fixed effects, IV, i(), and
  multiple-estimation operators.
- [Inference](references/inference.md) — vcov, clustering, HAC, weights, and SSC.
- [Reporting](references/reporting.md) — tidy results, tables, plots, and IV
  diagnostics.
- [Specialized estimators](references/specialized-estimators.md) — GLM, quantile,
  DiD/event-study, and multiple testing.
- [Demeaners](references/demeaners.md) — choose MAP or LSMR backend and reuse
  preconditioners.
- [Troubleshooting](references/troubleshooting.md) — formula, stored-data,
  optional-dependency, and unsupported-combination failures.
