---
name: pyfixest
description: Write, debug and port regression code that uses pyfixest (`import pyfixest as pf`) - feols, fepois, feglm and quantreg with fixed effects and IV, formula syntax, standard errors, post-estimation, multiple estimation and etable tables. Use when code imports pyfixest, when fitting fixed-effects regressions in Python, or when translating R fixest, Stata reghdfe or statsmodels code to pyfixest.
---

# pyfixest

Fixed-effects regression for Python, on pandas or polars DataFrames. Every
example below runs as written, once the optional packages named in comments are
installed; later blocks reuse `pf`, `data` and `fit` from the quick start.

## Reference files

The `references/` folder next to this file ships with the package, so it
matches the installed version. Open a file when this page is not enough:

- `references/formula-syntax.md`: every formula operator, `i()` arguments, varying slopes, IV, multiple estimation.
- `references/inference.md`: each `vcov` option and the models that support it, HAC, `pf.ssc()`, bootstrap, multiple testing.
- `references/coming-from-fixest.md`: R `fixest` spellings mapped to pyfixest, and behaviour that differs; for porting R code.
- `references/migrating-to-0.70.md`: deprecated or removed spellings and attributes; for pre-0.70 code or a `PyfixestDeprecationWarning`.

## Quick start

```python
import pyfixest as pf

data = pf.get_data()  # example data: Y, Y2, X1, X2, Z1, Z2, f1, f2, f3, group_id, weights
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
fit.summary()  # prints; returns None
fit.tidy()  # DataFrame: Estimate, Std. Error, t value, Pr(>|t|), 2.5%, 97.5%
fit.coef()["X1"]  # coef(), se(), tstat(), pvalue() return Series by coefficient
pf.etable([fit], type="md")  # prints a Markdown table
```

Rows with missing values and observations in singleton fixed-effect groups
are dropped (with a warning); the sample size is `fit.sample_info.n_obs`.

## Estimation functions

| Function | Model | Supports |
|---|---|---|
| `pf.feols(fml, data, vcov=None, weights=None, ...)` | OLS, WLS, 2SLS | fixed effects, IV, varying slopes, weights |
| `pf.fepois(fml, data, vcov=None, offset=None, ...)` | Poisson | fixed effects, varying slopes, weights, offset |
| `pf.feglm(fml, data, family, vcov=None, ...)` | `family`: `"logit"`, `"probit"`, `"gaussian"`, `"poisson"` | fixed effects, varying slopes, weights; `offset` only for `"poisson"` |
| `pf.quantreg(fml, data, quantile=0.5, vcov="nid", ...)` | quantile regression (experimental) | no fixed effects, no IV |

Each returns one fitted model, or a `FixestMulti` when the formula, `split`,
`fsplit` or a list `quantile=[0.25, 0.5]` asks for several models.
`weights="w"` names a column (`weights_type="fweights"` for frequency
weights). The demeaner is `demeaner=pf.LsmrDemeaner()` (default; tolerances
`fixef_atol`, `fixef_btol`) or `pf.MapDemeaner(fixef_tol=1e-8)`.

## Formula syntax

```python
formulas = [
    "Y ~ X1 + X2",  # intercept is implicit; "Y ~ 0 + X1" drops it
    "Y ~ X1 | f1 + f2",  # fixed effects after |
    "Y ~ X1 | f1:f2",  # one fixed effect per f1-by-f2 combination
    "Y ~ X1 | f1[X2]",  # f1 fixed effects plus one X2 slope per f1 level
    "Y ~ X1 + [X2 ~ Z1 + Z2] | f1",  # IV: X2 is endogenous, Z1 and Z2 instrument it
    "Y ~ X1 + C(f3)",  # dummies for a categorical variable
    "Y ~ i(f1, ref=0)",  # dummies with reference level 0, named "f1::1.0", ...
    "Y ~ i(f1, X1)",  # one X1 slope per f1 level (X1 is numeric)
    "Y ~ i(f1, C(f2), ref2=1)",  # f1-by-f2 dummies; C() makes f2 a factor
    "Y ~ X1 * X2 + I(X2**2) + np.exp(X2)",  # X1 + X2 + X1:X2; ** for powers; numpy as np
]
for fml in formulas:
    pf.feols(fml, data)
```

IV allows one endogenous regressor and only `feols`. Multiple estimation:
`sw(X2, Z1)` fits one model per term, `csw(X1, X2)` adds terms cumulatively,
`sw0`/`csw0` also fit the model without them, and `mvsw(X1, X2)` fits every
subset. All five work among the regressors (`Y ~ csw(X1, X2)`) and in the
fixed-effects part (`Y ~ X1 | csw0(f1, f2)`); on the left-hand side use only
`sw` (`sw(Y, Y2) ~ X1`). `split="f1"` fits one model per value of `f1`;
`fsplit="f1"` adds the full sample first.

## Inference

The default is `vcov="iid"` for every model, with or without fixed effects
(`"nid"` for `quantreg`). Pass `vcov` explicitly.

```python
pf.feols("Y ~ X1", data, vcov="hetero")  # "HC1" is the same; "HC2"/"HC3" need no FE and no IV
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"})  # cluster-robust
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1 + f2"})  # multiway; {"CRV3": "f1"} for the jackknife
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(G_adj=False))  # small-sample correction

panel = pf.get_worker_panel()  # one row per worker_id and year
hac = {"time_id": "year", "panel_id": "worker_id", "lag": 2}
pf.feols("log_wage ~ tenure | worker_id", panel, vcov="NW", vcov_kwargs=hac)  # or "DK"

fit.vcov("hetero").summary()  # recomputes inference in place and returns the fit
fit.vcov({"CRV1": "f1"})
```

Cluster variables must have no missing values in the estimation sample.
`fit.vcov()` fails on fits made with `lean=True` or `store_data=False`.
`CRV3` is not available for IV, varying-slope or logit, probit and gaussian
fits. Support by model, HAC options and `pf.ssc()`: `references/inference.md`.

## Post-estimation

```python
import numpy as np

fit.confint(alpha=0.10)  # DataFrame of bounds
fit.resid()  # ndarray
fit.predict(newdata=data.head())  # ndarray; unseen fixed-effect levels give NaN
fe = fit.fixef()  # DataFrame with columns variable, code, level, coefficient
fe[fe["variable"] == "f1"]
fit.wald_test(R=np.array([[1.0, -1.0]]), q=np.array([0.0]), distribution="chi2")  # WaldTest
fit.wildboottest(reps=999, param="X1", cluster="f1", seed=1)  # needs pip install wildboottest; returns a Series
pf.feols("Y ~ X1 + X2", data).decompose(decomp_var="X1", x1_vars=["X2"], reps=50, seed=1)

counts = pf.get_data(model="Fepois")  # count outcome
pois = pf.fepois("Y ~ X1 | f1", counts, vcov={"CRV1": "f1"})
pois.predict(type="response")  # expected counts; the default type="link" is the linear predictor

iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)
iv.first_stage.model.tidy()  # first_stage is a component holding the first-stage fit
iv.first_stage.diagnostics.f_stat
pf.etable([iv.first_stage.model, iv], type="md")
```

`fit.resid()` and `fit.predict()` cover only the estimation sample, so
`data["r"] = fit.resid()` fails when rows were dropped;
`fit.sample_info.dropped_row_index` holds the dropped rows.

Not every model supports every method: `fit.capabilities` lists what a fit
supports, and an unsupported call raises `NotImplementedError`. IV and
varying-slope fits have no `predict()` or `fixef()`; `wildboottest()` and
`decompose()` need a `feols` fit without IV or varying slopes. Randomization
inference, `ccv()` and multiple-testing corrections: `references/inference.md`.

Also available: `pf.coefplot([fit])`, `pf.iplot(fit)` for `i()` terms, and
`pf.event_study(..., estimator="twfe")` (or `"did2s"`, `"saturated"`),
`pf.did2s`, `pf.lpdid` for difference-in-differences. Their signatures are
`pf.event_study(data, yname, idname, tname, gname, estimator="twfe")` and
`pf.did2s(data, yname, first_stage="~ 0 | unit + year", second_stage="~ i(rel_year, ref=-1.0)", treatment="treat", cluster="unit")`,
where the stage formulas start with `~` and have no left-hand side.

## Fitted-model components

Results live in documented components. Most private attributes from 0.60,
such as `fit._se`, now raise `AttributeError`; replacements are in
`references/migrating-to-0.70.md`.

```python
fit.coeftable.se  # ndarray; also estimate, tstat, pvalue, conf_int (2 x k)
fit.variance_covariance.vcov  # k x k ndarray; .spec.vcov_type is e.g. "CRV"
fit.fitstat.r2, fit.fitstat.r2_within  # also rmse, adj_r2, deviance, loglik, pseudo_r2; NaN if n/a
fit.sample_info.n_obs, fit.sample_info.dropped_by_stage  # dropped rows by reason
fit.model.depvar, fit.model.fixed_effects  # GLMs add fit.convergence; IV fits fit.first_stage
```

## Fitted models from multiple estimation

```python
multi = pf.feols("Y ~ X1 | csw0(f1, f2)", data)
list(multi.all_fitted_models)  # ['Y ~ 1 + X1', 'Y ~ X1 | f1', 'Y ~ X1 | f1 + f2']
multi.all_fitted_models["Y ~ X1 | f1"]  # keys are canonical: print them, do not guess
multi.fetch_model(0, print_fml=False)  # by position; multi.to_list() gives all
multi.tidy()  # one DataFrame indexed by (fml, Coefficient)
multi.vcov("hetero")  # changes inference for every model
pf.etable(multi, type="md")
```

## Regression tables

```python
fit2 = pf.feols("Y ~ X1 | f1", data)
pf.etable(
    [fit, fit2],  # a list of fits, one fit, or a FixestMulti
    type="df",  # "gt" (default), "md", "df", "tex", "html", "typst", "docx"
    keep=["X1"],  # regex patterns; drop= works the same way
    labels={"X1": "Treatment"},
    felabels={"f1": "Firm FE", "f2": "Year FE"},
    coef_fmt="b (se)",
    model_heads=["Two-way FE", "One-way FE"],
    notes="Clustered by f1.",
)
```

`"gt"` returns a great_tables object, `"md"` prints and returns None, `"df"` a
DataFrame, `"tex"`, `"html"` and `"typst"` strings, `"docx"` a python-docx
Document. `file_name=` saves `"gt"`, `"tex"`, `"typst"` and `"docx"` tables.

## Coming from R fixest, Stata or statsmodels

The full R lookup table is `references/coming-from-fixest.md`; old pyfixest
spellings are in `references/migrating-to-0.70.md`. The most common slips:

- `^` is not a power: write `I(X1**2)`, not `X1^2` or `I(X1^2)`.
- `factor(x)`, Stata `i.x`: write `C(x)` or `i(x)`. `i(f, k)` with a numeric
  `k` gives one `k` slope per `f` level; `i(f, C(k))` gives f-by-k dummies.
- `| f1^f2`, `Y ~ X1 | f1 | X2 ~ Z1` and `Y1 + Y2 ~ X1` are deprecated:
  write `| f1:f2`, `Y ~ X1 + [X2 ~ Z1] | f1` and `sw(Y1, Y2) ~ X1`.
- There are no `cluster=` or `se=` arguments and no `~` in arguments:
  `vcov={"CRV1": "f1"}`, `vcov={"CRV1": "f1 + f2"}` (not `"twoway"`),
  `vcov="hetero"` (Stata `robust`), `weights="w"`, `split="k"`.
- Stata `reghdfe y x, absorb(f1 f2) vce(cluster f1)` is
  `pf.feols("y ~ x | f1 + f2", data, vcov={"CRV1": "f1"})`.
- `pf.etable(fit1, fit2)` fails because the second argument is `type`: pass a
  list, `pf.etable([fit1, fit2])`.
- From pyfixest 0.60 or earlier: `fixef_tol=`, `fixef_maxiter=` and
  `demeaner_backend=` are gone, use `demeaner=`; `pf.ssc(adj=, fixef_k=,
  cluster_adj=, cluster_df=)` became `k_adj, k_fixef, G_adj, G_df`;
  `confint(joint=True)` became `inference_type="simult"`; `pf.dtable` is
  removed; `wald_test()` returns a `WaldTest` with `.stat` and `.pvalue`, not
  a Series.
- `summary(fit, vcov=...)` becomes `fit.vcov(...).summary()`.
- statsmodels: no `.fit()` step; `fit.params`, `fit.bse`, `fit.nobs`,
  `fit.rsquared` are the methods `fit.coef()`, `fit.se()` and the fields
  `fit.sample_info.n_obs`, `fit.fitstat.r2`.

## Where to look next

- Docs: https://pyfixest.org; API: https://pyfixest.org/reference/index.html; index for agents: https://pyfixest.org/llms.txt
- Tutorials (OLS, IV, tables, DiD): https://pyfixest.org/tutorials/index.html
- Stata to pyfixest: https://pyfixest.org/how-to/stata-2-pyfixest.html
- If fixed-effects estimation is slow: the default demeaner is chosen to be fast across a variety of fixed-effects structures, but changing its parameters or switching demeaner can sometimes help further, and preconditioners can be stored across regressions and sessions: https://pyfixest.org/how-to/demeaner-backends.html
