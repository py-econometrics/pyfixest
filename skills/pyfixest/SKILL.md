---
name: pyfixest
description: "Use for fixest-style econometrics in Python with PyFixest: formulas, OLS/WLS/IV, high-dimensional fixed effects, GLMs, quantile regression, inference, reporting, DiD, demeaning backends, or troubleshooting. Read the bundled version-matched docs before guessing syntax, covariance options, or result APIs."
---

# PyFixest

Use the documentation installed with the package before relying on memory. Start
at `pyfixest/docs/index.md`; its content matches the installed PyFixest version.
Use <https://pyfixest.org/llms.txt> when the local corpus is unavailable.

## Workflow

1. Identify the estimator, formula features, inference method, and output the
   user needs.
2. Read the relevant reference below and follow its links into the bundled
   documentation.
3. Check the installed function signature or API reference for the installed
   version before writing code.
4. Prefer public `pyfixest` functions and result methods. Do not depend on
   underscore-prefixed model state.
5. State unsupported estimator/inference combinations instead of silently
   substituting a method.
6. Use small, reproducible examples and preserve the user's input data.

## Focused references

- [Core API and result objects](references/core-api.md)
- [Formula syntax and multiple estimation](references/formula-syntax.md)
- [Standard errors and inference](references/inference.md)
- [Tables, plots, and model reporting](references/reporting.md)
- [Specialized estimators and causal workflows](references/specialized-estimators.md)
- [Demeaning backends](references/demeaners.md)
- [Troubleshooting and model state](references/troubleshooting.md)
