---
name: pyfixest
description: Write, debug and port regression code that uses pyfixest (`import pyfixest as pf`) - feols, fepois, feglm and quantreg with fixed effects and IV, formula syntax, standard errors, post-estimation, multiple estimation and etable tables. Use when code imports pyfixest, when fitting fixed-effects regressions in Python, or when translating R fixest, Stata reghdfe or statsmodels code to pyfixest.
---

# pyfixest

Fixed-effects regression for Python, on pandas or polars DataFrames. Every
example below runs as written; later blocks reuse `pf`, `data` and `fit` from
the quick start.

## Quick start

```python
import pyfixest as pf

data = pf.get_data()  # example data: Y, Y2, X1, X2, Z1, Z2, f1, f2, f3, weights
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
fit.summary()  # prints; returns None
fit.tidy()  # DataFrame: Estimate, Std. Error, t value, Pr(>|t|), 2.5%, 97.5%
fit.coef()["X1"]  # coef(), se(), tstat(), pvalue() return Series by coefficient
pf.etable([fit], type="md")  # prints a Markdown table
```

Rows with missing values in model variables are dropped, and so are
observations in singleton fixed-effect groups (with a warning). The estimation
sample size is `fit.sample_info.n_obs`.

## Estimation functions

| Function | Model | Supports |
|---|---|---|
| `pf.feols(fml, data, vcov=None, weights=None, ...)` | OLS, WLS, 2SLS | fixed effects, IV, varying slopes, weights |
| `pf.fepois(fml, data, vcov=None, offset=None, ...)` | Poisson | fixed effects, varying slopes, weights, offset |
| `pf.feglm(fml, data, family, vcov=None, ...)` | `family`: `"logit"`, `"probit"`, `"gaussian"`, `"poisson"` | fixed effects, varying slopes, weights, offset |
| `pf.quantreg(fml, data, quantile=0.5, vcov="nid", ...)` | quantile regression (experimental) | no fixed effects, no IV |

Each returns one fitted model, or a `FixestMulti` when the formula, `split`,
`fsplit` or a list `quantile=[0.25, 0.5]` asks for several models.
`weights="w"` names a column; `weights_type="fweights"` makes them frequency
weights (default `"aweights"`). Fixed effects are projected out by
`demeaner=pf.LsmrDemeaner()` (the default; tolerances `fixef_atol`,
`fixef_btol`) or `pf.MapDemeaner(fixef_tol=1e-8)` (alternating projections).

## Formula syntax

```python
formulas = [
    "Y ~ X1 + X2",  # intercept is implicit; "Y ~ 0 + X1" drops it
    "Y ~ X1 | f1 + f2",  # fixed effects after |
    "Y ~ X1 | f1:f2",  # one fixed effect per f1-by-f2 combination
    "Y ~ X1 | f1[X2]",  # f1 fixed effects plus one X2 slope per f1 level
    "Y ~ X1 | f1[[X2]]",  # the X2 slopes per f1 level only
    "Y ~ X1 + [X2 ~ Z1 + Z2] | f1",  # IV: X2 is endogenous, Z1 and Z2 instrument it
    "Y ~ X1 + C(f3)",  # dummies for a categorical variable
    "Y ~ i(f1, ref=0)",  # dummies with reference level 0, named "f1::1.0", ...
    "Y ~ i(f1, X1)",  # one X1 slope per f1 level (X1 is numeric)
    "Y ~ i(f1, C(f2), ref2=1)",  # f1-by-f2 dummies; C() makes f2 a factor
    "Y ~ i(f1, bin={'low': [0, 1, 2]}, ref='low')",  # merge levels into one
    "Y ~ X1 + I(X1**2) + np.exp(X2)",  # arithmetic: ** for powers, numpy as np
    "Y ~ X1 * X2",  # X1 + X2 + X1:X2
]
for fml in formulas:
    pf.feols(fml, data)
```

The slopes of `f1[X2]` are absorbed like fixed effects and are not reported as
coefficients; they need the default demeaner. IV allows one endogenous
variable and only `feols`.

Multiple estimation operators expand one formula into several models:
`sw(X2, Z1)` fits one model per term, `csw(X1, X2)` adds terms cumulatively,
`sw0` and `csw0` first fit the model without them, and `mvsw(X1, X2)` fits
every subset. They work on the left-hand side (`sw(Y, Y2) ~ X1`), among the
covariates (`Y ~ csw(X1, X2)`) and in the fixed-effects part
(`Y ~ X1 | csw0(f1, f2)`). `split="f1"` fits one model per value of the column
`f1`; `fsplit="f1"` adds the full sample first.

## Inference

The default is `vcov="iid"` for every model, with or without fixed effects
(`"nid"` for `quantreg`). Pass `vcov` explicitly.

```python
pf.feols("Y ~ X1", data, vcov="hetero")  # "HC1" is the same; "HC2"/"HC3" need no FE and no IV
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"})  # cluster-robust
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1 + f2"})  # multiway clustering
pf.feols("Y ~ X1 | f1", data, vcov={"CRV3": "f1"})  # leave-one-cluster-out jackknife
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(G_adj=False))  # small-sample correction

panel = pf.get_worker_panel()  # one row per worker_id and year
hac = {"time_id": "year", "panel_id": "worker_id", "lag": 2}  # omit panel_id for one time series
pf.feols("log_wage ~ tenure | worker_id", panel, vcov="NW", vcov_kwargs=hac)  # Newey-West
pf.feols("log_wage ~ tenure | worker_id", panel, vcov="DK", vcov_kwargs=hac)  # Driscoll-Kraay

fit.vcov("hetero").summary()  # recomputes inference in place and returns the fit
fit.vcov({"CRV1": "f1"})
```

`pf.ssc()` defaults to `k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min"`.
A cluster variable must not be missing in the estimation sample
(`NanInClusterVarError`); drop those rows first. To cluster on an interaction,
add it as a column. `fit.vcov()` needs the data, so it fails on fits made with
`lean=True` or `store_data=False`.

## Post-estimation

```python
import numpy as np

fit.confint(alpha=0.10)  # DataFrame of bounds
fit.resid()  # ndarray
fit.predict(newdata=data.head())  # ndarray; unseen fixed-effect levels give NaN
fe = fit.fixef()  # DataFrame with columns variable, code, level, coefficient
fe[fe["variable"] == "f1"]
fit.wald_test(R=np.array([[1.0, -1.0]]), q=np.array([0.0]), distribution="chi2")  # WaldTest
fit.wildboottest(reps=999, param="X1", cluster="f1", seed=1)  # Series with bootstrap p-value
fit.ritest(resampvar="X1", reps=199, rng=np.random.default_rng(1))  # randomization inference
pf.feols("Y ~ X1 + X2", data).decompose(decomp_var="X1", x1_vars=["X2"], reps=50, seed=1)

counts = pf.get_data(model="Fepois")  # count outcome
pois = pf.fepois("Y ~ X1 | f1", counts, vcov={"CRV1": "f1"})
pois.predict(type="response")  # expected counts; the default type="link" is the linear predictor

iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)
iv.first_stage.model.tidy()  # first_stage is a component holding the first-stage fit
iv.first_stage.diagnostics.f_stat
pf.etable([iv.first_stage.model, iv], type="md")
```

Not every model supports every method. `fit.capabilities` lists what a fit
supports, and an unsupported call raises `NotImplementedError` naming the
missing capability. Common gaps: IV fits have no `predict()` or `fixef()`;
varying-slope fits have no `predict()`; `wildboottest()` and the Gelbach
`decompose()` need a `feols` fit without IV or varying slopes, and
`wildboottest()` also rejects weights; `ccv()` rejects fixed effects.

Also available: `pf.coefplot([fit])`, `pf.iplot(fit)` for `i()` terms,
`pf.bonferroni`, `pf.rwolf`, `pf.wyoung` for multiple testing, and
`pf.event_study(..., estimator="twfe")` (or `"did2s"`, `"saturated"`),
`pf.did2s`, `pf.lpdid` for difference-in-differences.

## Fitted-model components

Results live in documented components. Most private attributes from 0.60,
such as `fit._se`, now raise `AttributeError`; the migration table is at
https://pyfixest.org/how-to/migrate-fit-attributes.html

```python
fit.coeftable.se  # ndarray; also estimate, tstat, pvalue, conf_int (2 x k)
fit.variance_covariance.vcov  # k x k ndarray; .spec.vcov_type is e.g. "CRV"
fit.fitstat.r2, fit.fitstat.r2_within  # also rmse, adj_r2, deviance, loglik, pseudo_r2; NaN if n/a
fit.sample_info.n_obs, fit.sample_info.dropped_by_stage  # dropped rows by reason
fit.model.depvar, fit.model.fixed_effects
```

Coefficient names are `fit.coef().index`. GLM fits also carry
`fit.convergence`; IV fits carry `fit.first_stage`.

## Multiple estimation results

```python
multi = pf.feols("Y ~ X1 | csw0(f1, f2)", data)
list(multi.all_fitted_models)  # ['Y ~ 1 + X1', 'Y ~ X1 | f1', 'Y ~ X1 | f1 + f2']
multi.all_fitted_models["Y ~ X1 | f1"]
multi.fetch_model(0, print_fml=False)  # by position
multi.to_list()
multi.tidy()  # one DataFrame indexed by (fml, Coefficient)
multi.vcov("hetero")  # changes inference for every model
pf.etable(multi, type="md")
```

Keys are canonical formulas: models without fixed effects gain `1 +`, and
split samples end in `(Sample: f1 = 0.0)`. Print the keys rather than guess.

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

`"gt"` returns a great_tables object for notebooks, `"md"` prints and returns
None, `"df"` returns a DataFrame, `"tex"`, `"html"` and `"typst"` return
strings, and `"docx"` returns a python-docx Document. `file_name=` saves
`"gt"`, `"tex"`, `"typst"` and `"docx"` tables.

## Coming from R fixest, Stata or statsmodels

- `X1^2` and `I(X1^2)` do not square: write `I(X1**2)`.
- `factor(x)`, `as.factor(x)`, Stata `i.x`: write `C(x)` or `i(x)`.
- `i(f, k)` with a numeric `k` gives one `k` slope per `f` level; for
  f-by-k dummies write `i(f, C(k))`.
- `| f1^f2` is deprecated: write `| f1:f2`.
- `Y ~ X1 | f1 | X2 ~ Z1` is deprecated: write `Y ~ X1 + [X2 ~ Z1] | f1`.
- `c(Y1, Y2) ~ X1` fails and `Y1 + Y2 ~ X1` is deprecated: write `sw(Y1, Y2) ~ X1`.
- There are no `cluster=` or `se=` arguments and no `~` in arguments:
  `vcov={"CRV1": "f1"}`, `vcov={"CRV1": "f1 + f2"}` (not `"twoway"`),
  `vcov="hetero"` (Stata `robust`), `weights="w"`, `split="k"`.
- Stata `reghdfe y x, absorb(f1 f2) vce(cluster f1)` is
  `pf.feols("y ~ x | f1 + f2", data, vcov={"CRV1": "f1"})`.
- No panel operators (`l()`, `d()`, `L.x`, `panel.id`): compute lags in
  pandas, e.g. `df.sort_values("year").groupby("unit")["x"].shift(1)`.
- No formula macros (`..ctrl`, `xpd()`): build the formula string in Python.
- `pf.etable(fit1, fit2)` fails because the second argument is `type`: pass a
  list, `pf.etable([fit1, fit2])`.
- `summary(fit, vcov=...)` becomes `fit.vcov(...).summary()`.
- `fit.fixef()` returns a DataFrame, not a dict: `fe[fe["variable"] == "f1"]`.
- `fit.first_stage` is a component, not a method: `fit.first_stage.model`.
- `fit.se()`, `fit.coef()` and `fit.resid()` are methods; `fit.coeftable.se`
  is an array.
- statsmodels: no `.fit()` step; `fit.params`, `fit.bse`, `fit.nobs`,
  `fit.rsquared` are `fit.coef()`, `fit.se()`, `fit.sample_info.n_obs`,
  `fit.fitstat.r2`.
- Fixed-effect columns need a concrete dtype (integer, float, `"string"`,
  `"category"`), not `object`: convert with `.astype("string")`. Their names
  must be Python identifiers: rename `firm.id` to `firm_id`.

## Where to look next

- Docs: https://pyfixest.org; API reference: https://pyfixest.org/reference/index.html
- Index of all pages for agents: https://pyfixest.org/llms.txt
- Formula syntax: https://pyfixest.org/tutorials/formula-syntax.html
- OLS with fixed effects: https://pyfixest.org/tutorials/ols-fixed-effects.html
- IV: https://pyfixest.org/tutorials/instrumental-variables.html
- Tables: https://pyfixest.org/tutorials/regression-tables.html
- Difference-in-differences: https://pyfixest.org/tutorials/difference-in-differences.html
- Stata to pyfixest: https://pyfixest.org/how-to/stata-2-pyfixest.html
- If fixed-effects estimation is slow: the default demeaner is chosen to be fast across a variety of fixed-effects structures, but changing its parameters or switching demeaner can sometimes help further, and preconditioners can be stored across regressions and sessions: https://pyfixest.org/how-to/demeaner-backends.html
