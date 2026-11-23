# Inference reference

Standard errors, small-sample corrections, and resampling and
multiple-testing procedures. Every example runs as written; later blocks reuse
earlier names. API reference: https://pyfixest.org/reference/index.html;
small-sample details: https://pyfixest.org/explanation/ssc.html

```python
import numpy as np
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data, vcov={"CRV1": "f1"})
```

## `vcov` values

The default is `vcov="iid"` for `feols`, `fepois` and `feglm`, with or
without fixed effects, and `"nid"` for `quantreg`. There are no `cluster=` or
`se=` arguments.

| `vcov=` | Variance estimator |
|---|---|
| `"iid"` | homoskedastic |
| `"hetero"` or `"HC1"` | heteroskedasticity-robust (same estimator) |
| `"HC2"`, `"HC3"` | leverage-adjusted robust |
| `{"CRV1": "f1"}` | cluster-robust by column `f1` |
| `{"CRV1": "f1 + f2"}` | multiway clustering; any number of columns joined by `+` |
| `{"CRV3": "f1"}` | leave-one-cluster-out jackknife (refits the model per cluster) |
| `"NW"` | Newey-West HAC; needs `vcov_kwargs` |
| `"DK"` | Driscoll-Kraay HAC; needs `vcov_kwargs` |
| `"nid"` | `quantreg` only: Hendricks-Koenker sandwich |

Which model supports which (unsupported calls raise
`VcovTypeNotSupportedError` or `NotImplementedError`):

| | `feols` | `feols` IV | `fepois` | `feglm` | `quantreg` |
|---|---|---|---|---|---|
| `iid`, `hetero`, `HC1` | yes | yes | yes | yes | yes |
| `HC2`, `HC3` | no fixed effects | no | no fixed effects | no fixed effects | same as `hetero` |
| `CRV1`, one cluster | yes | yes | yes | yes | yes |
| `CRV1`, multiway | yes | yes | yes | yes | no |
| `CRV3` | yes | no | yes | `family="poisson"` only | no |
| `NW`, `DK` | yes | yes | yes | yes | no |

Varying-slope models (`| f1[X2]`) support neither `CRV3` nor HAC. To check a
fitted model, read `fit.capabilities.crv3_inference`,
`.hac_inference` and `.multiway_clustering`.

## Clustering rules

- Pass a string: `{"CRV1": "f1 + f2"}`. A list, `"twoway"`, `"~f1"` and
  `"cluster"` are rejected.
- The cluster column must have no missing values in the estimation sample
  (`NanInClusterVarError`); drop those rows before fitting.
- To cluster on an interaction, add it as a column and cluster on that:

```python
data["f1_f2"] = data["f1"].astype(str) + "_" + data["f2"].astype(str)
pf.feols("Y ~ X1", data.dropna(subset=["f1"]), vcov={"CRV1": "f1_f2"})
```

## HAC: Newey-West and Driscoll-Kraay

`vcov_kwargs` accepts exactly the keys `time_id`, `panel_id` and `lag`.
Consecutive time values are treated as one period apart.

```python
panel = pf.get_worker_panel()  # worker_id, firm_id, year, log_wage, tenure, ...
hac = {"time_id": "year", "panel_id": "worker_id", "lag": 2}
pf.feols("log_wage ~ tenure | worker_id", panel, vcov="NW", vcov_kwargs=hac)
pf.feols("log_wage ~ tenure | worker_id", panel, vcov="DK", vcov_kwargs=hac)

series = panel.groupby("year", as_index=False)[["log_wage", "tenure"]].mean()
pf.feols("log_wage ~ tenure", series, vcov="NW", vcov_kwargs={"time_id": "year", "lag": 1})
```

- Panel NW and DK need `time_id` and `panel_id`; without `lag` they use
  `floor(T ** 0.25)` lags, with `T` the number of distinct periods.
- Time-series NW (no `panel_id`) needs `lag` and unique time values.

## Small-sample corrections: `pf.ssc()`

```python
pf.ssc()  # Ssc(k_adj=True, k_fixef='nonnested', G_adj=True, G_df='min')
pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False, G_adj=False))
```

| Argument | Values | Effect |
|---|---|---|
| `k_adj` | `True`, `False` | multiply by `(N - 1) / (N - k)` |
| `k_fixef` | `"nonnested"`, `"full"`, `"none"` | how fixed effects count in `k`; `"nonnested"` skips effects nested in a cluster |
| `G_adj` | `True`, `False` | multiply cluster-robust estimates by `G / (G - 1)` |
| `G_df` | `"min"`, `"conventional"` | multiway: one `G` (the smallest) or each dimension's own `G` |

- `ssc=` is set when fitting; `fit.vcov(...)` keeps the fit's `ssc`. Refit to
  change it.
- A dict such as `{"k_adj": False}` also works. The old names `adj`,
  `fixef_k`, `cluster_adj` and `cluster_df` emit a `DeprecationWarning`.
- `feols` p-values use a t distribution with `N - k` degrees of freedom, or
  `G - 1` when clustered (the smallest `G` for multiway clustering). GLMs use
  the normal distribution. `fit.variance_covariance.df_t` holds the value.

## Changing inference after fitting

```python
fit.vcov("hetero").summary()  # recomputes in place and returns the fit
fit.vcov({"CRV1": "f1 + f2"})
fit.variance_covariance.spec  # VcovSpec(vcov_type='CRV', vcov_type_detail='CRV1', ...)
fit.variance_covariance.G  # cluster counts per dimension
multi = pf.feols("Y ~ csw(X1, X2)", data, vcov="hetero")
multi.vcov({"CRV1": "group_id"})  # every model at once
```

- `fit.vcov()` mutates `fit`; `coef()`, `se()`, `tidy()`, `summary()` and
  `etable()` then show the new inference.
- It needs the stored data: fits made with `lean=True` or `store_data=False`
  raise `MissingModelDataError`. Choose `vcov=` when fitting instead.
- `fit.vcov(..., data=df)` raises `ValueError`; there is no `data` argument.
- There is no `summary(vcov=...)`; call `fit.vcov(...).summary()`.

## Confidence intervals and Wald tests

```python
fit.confint(alpha=0.10)  # 90% intervals, DataFrame
fit.tidy(alpha=0.10)  # estimates with 90% intervals
fit.confint(inference_type="simult", seed=1)  # simultaneous bands; seed makes them reproducible
fit.wald_test()  # joint test that all coefficients are zero (F)
fit.wald_test(R=np.array([[1.0, -1.0]]), q=np.array([0.0]), distribution="chi2")  # X1 = X2
```

`wald_test()` returns a `WaldTest` (`stat`, `pvalue`, `df1`, `df2`,
`distribution`). For any `R` other than the identity, or a nonzero `q`, pass
`distribution="chi2"`; with the default `"F"` pyfixest switches to chi2 and
warns. `confint(joint=True)` is deprecated; use `inference_type="simult"`.

Anytime-valid inference (SAVI) works for `feols` without IV, fixed effects or
weights, and with an iid or heteroskedasticity-robust variance estimator:

```python
ols = pf.feols("Y ~ X1 + X2", data, vcov="hetero")
ols.confint(inference_type="savi")
ols.evalue()  # Series of e-values
ols.pvalue_savi()
```

## Wild cluster bootstrap

```python
fit.wildboottest(reps=999, param="X1", cluster="f1", seed=1)  # needs pip install wildboottest
```

Returns a Series with `t value`, `Pr(>|t|)`, `bootstrap_type` and
`inference`. `param` is required. Without `cluster=`, it uses the fit's
cluster variable, or a heteroskedastic wild bootstrap if the fit is not
clustered. Other arguments: `weights_type="rademacher"`, `impose_null=True`,
`bootstrap_type="11"`. Only unweighted `feols` fits without IV or varying
slopes support it.

## Randomization inference and causal cluster variance

```python
fit.ritest(resampvar="X1", reps=199, rng=np.random.default_rng(1))  # H0: X1 = 0
fit.ritest(resampvar="X1=-0.9", reps=199, type="randomization-t", rng=np.random.default_rng(1))
```

`ritest()` permutes the treatment column and returns a Series with the
p-value `Pr(>|t|)` and its Monte Carlo interval. `type="randomization-c"` (the
default) is faster; `"randomization-t"` uses the t-statistic. With
`cluster="g"`, the treatment must be constant within each cluster, and
cluster-level values are permuted. It supports `feols` without IV and
`fepois`.

```python
small = data.dropna().assign(D=lambda d: (d["X1"] > 0).astype(int))
pf.feols("Y ~ D", small).ccv(treatment="D", cluster="group_id", seed=1)
```

`ccv()` needs a 0/1 treatment and a `feols` fit without fixed effects or
weights.

## Multiple-testing adjustments

```python
models = pf.feols("sw(Y, Y2) ~ X1 | f1", data, vcov={"CRV1": "f1"})
pf.bonferroni(models, param="X1")
pf.rwolf(models, param="X1", reps=199, seed=1)  # Romano-Wolf, wild bootstrap
pf.wyoung(models, param="X1", reps=199, seed=1)  # Westfall-Young
```

Each takes a `FixestMulti` or a list of fits and returns a DataFrame with one
column per model and an added row (`Bonferroni Pr(>|t|)`, `RW Pr(>|t|)`,
`WY Pr(>|t|)`). `rwolf()` and `wyoung()` also take
`sampling_method="ri"` for randomization inference.
