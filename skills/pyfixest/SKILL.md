---
name: pyfixest
description: Routes coding agents into PyFixest's bundled, version-matched documentation for fixed-effects regression in Python (feols, fepois, feglm, quantreg, difference-in-differences estimators), standard errors and inference, and regression tables. Use when code imports pyfixest or a task mentions fixed effects, clustered standard errors, IV, event studies, or fixest-style formulas.
---

# PyFixest

The installed package ships its own documentation. Locate it once:

    python -c "import importlib.resources as r; print(r.files('pyfixest') / 'docs')"

Read `cheatsheet.llms.md` in that directory first (one page: syntax, standard
errors, methods, and the table below). `llms.txt` lists every page. If
`pyfixest` is not importable, use https://pyfixest.org/cheatsheet.html and
https://pyfixest.org/skills.html instead.

## Where to go for each task

| Task | Start here | Then |
|---|---|---|
| Fit a first model | `getting-started.llms.md` | `tutorials/ols-fixed-effects.llms.md` |
| Formulas, fixed effects, interactions, multiple estimation | `tutorials/formula-syntax.llms.md` | `reference/estimation.api.feols.feols.llms.md` |
| Instrumental variables | `tutorials/instrumental-variables.llms.md` | `reference/estimation.api.feols.feols.llms.md` |
| Poisson or GLM | `tutorials/poisson-glm.llms.md` | `reference/estimation.api.fepois.fepois.llms.md`, `reference/estimation.api.feglm.feglm.llms.md` |
| Quantile regression | `tutorials/quantile-regression.llms.md` | `reference/estimation.api.quantreg.quantreg.llms.md` |
| Choose or change standard errors | `tutorials/standard-errors.llms.md` | `reference/estimation.models.feols_.Feols.vcov.llms.md` |
| Tables, summaries, coefficient plots | `tutorials/regression-tables.llms.md` | `reference/index.llms.md` |
| Difference-in-differences, event studies | `tutorials/difference-in-differences.llms.md` | `reference/index.llms.md#difference-in-differences` |
| Predictions, residuals, fixed-effect estimates | `reference/estimation.models.feols_.Feols.predict.llms.md` | `reference/estimation.models.feols_.Feols.resid.llms.md`, `reference/estimation.models.feols_.Feols.fixef.llms.md` |
| Hypothesis tests, decomposition | `reference/estimation.models.feols_.Feols.wald_test.llms.md` | `how-to/regression_decomposition.llms.md` |
| Multiple-hypothesis corrections | `reference/estimation.post_estimation.multcomp.rwolf.llms.md` | `reference/estimation.post_estimation.multcomp.wyoung.llms.md`, `reference/estimation.post_estimation.multcomp.bonferroni.llms.md` |
| Anytime-valid / sequential inference | `how-to/anytime-valid-inference.llms.md` | `reference/estimation.models.feols_.Feols.evalue.llms.md`, `reference/estimation.models.feols_.Feols.pvalue_savi.llms.md` |
| Marginal effects | `how-to/marginaleffects.llms.md` | `reference/index.llms.md` |
| AB tests with panel data | `how-to/panel_variance_reduction.llms.md` | `tutorials/difference-in-differences.llms.md` |
| Slow or failing fixed-effects fits | `explanation/difficult-fixed-effects.llms.md` | `how-to/demeaner-backends.llms.md` |
| Translate from R or Stata | `explanation/compare-fixest-pyfixest.llms.md` | `how-to/stata-2-pyfixest.llms.md` |
| Exact signatures | `reference/index.llms.md` | `llms.txt` |

## Core facts

```python
import pyfixest as pf
data = pf.get_data().dropna()
# fixed effects follow the first |, : interacts them (`^` is deprecated)
pf.feols("Y ~ X1 + X2 | f1 + f2", data=data)
pf.feols("Y ~ X1 | f1:f2", data=data)
# three-part IV: depvar ~ exogenous | fixed effects | endogenous ~ instruments
pf.feols("Y ~ X2 | f1 | X1 ~ Z1", data=data)  # newer spelling: Y ~ X2 + [X1 ~ Z1] | f1
# i(cat) expands a categorical, ref drops a level, i(cat, x) gives slopes on x
pf.feols("Y ~ i(f1, ref=1.0) + i(f1, X2)", data=data)
# vcov spellings; NW and DK additionally need vcov_kwargs with time metadata
pf.feols("Y ~ X1", data=data, vcov="iid")
pf.feols("Y ~ X1", data=data, vcov="hetero")
pf.feols("Y ~ X1", data=data, vcov="HC3")
pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1"})
pf.feols("Y ~ X1", data=data, vcov={"CRV3": "f1"})
# two-way clustering
pf.feols("Y ~ X1", data=data, vcov={"CRV1": "f1 + f2"})
pf.etable([pf.feols("Y ~ X1 | f1", data=data)], type="md", keep="X1")
```

## Workflow

1. Choose the closest task in the table above and open its narrative
   documentation.
2. Before choosing `vcov`, check the support-limits table in
   `tutorials/standard-errors.llms.md`; before writing a formula, check
   `tutorials/formula-syntax.llms.md`.
3. Check the installed API reference or function signature before writing code.
4. Search `llms.txt` only when the table above does not identify a suitable
   page.
5. Prefer public `pyfixest` functions and result methods; do not depend on
   underscore-prefixed state.
6. State unsupported combinations instead of silently substituting another
   estimator or inference method.
