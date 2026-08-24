# Standard Errors & Inference

Core Estimation

Choose and configure IID, heteroskedasticity-robust, clustered, HAC, bootstrap, and randomization-based inference.

Point estimates answer what the fitted model estimates. Standard errors and tests answer how uncertain those estimates are under a particular dependence or assignment model. Choose that model from the research design: changing `vcov` after seeing the results is not a substitute for deciding which observations may share shocks.

This guide uses a small cross-sectional dataset and a small simulated panel.

``` python
import numpy as np
import pandas as pd
import pyfixest as pf

data = pf.get_data(N=300, seed=123).dropna()

rng = np.random.default_rng(123)
n_units = 12
n_periods = 10
panel = pd.DataFrame(
    {
        "panel_id": np.repeat(np.arange(n_units), n_periods),
        "time_id": np.tile(np.arange(n_periods), n_units),
    }
)
panel["x"] = rng.normal(size=len(panel))
panel["y"] = (
    1
    + 0.5 * panel["x"]
    + np.repeat(rng.normal(size=n_units), n_periods)
    + rng.normal(size=len(panel))
)
```

## A short selection guide

| Dependence or design | PyFixest choice |
|----|----|
| Independent errors with constant variance | `vcov="iid"` |
| Independent errors with unknown variance | `vcov="hetero"`, `"HC1"`, `"HC2"`, or `"HC3"` |
| Errors correlated within one or two groups | `{"CRV1": "cluster"}` or `{"CRV1": "cluster1 + cluster2"}` |
| Few clusters and a cluster jackknife is appropriate | `{"CRV3": "cluster"}` |
| Serial correlation in a time series or panel | `vcov="NW"` with `vcov_kwargs` |
| Cross-sectional and serial dependence in a panel | `vcov="DK"` with `vcov_kwargs` |
| Few clusters and an unweighted linear model | `wildboottest()` |
| Known random assignment mechanism | `ritest()` |

These choices are not interchangeable. For example, heteroskedasticity-robust inference does not allow arbitrary correlation within firms, and clustering by firm does not by itself model common shocks across firms in the same year.

## IID and HC1–HC3

`vcov="iid"` assumes independent errors with a common variance. It is a useful benchmark when that assumption is credible.

HC estimators retain independence across observations but allow their variances to differ:

- `"hetero"` and `"HC1"` are aliases.
- `"HC2"` adjusts squared residuals using leverage.
- `"HC3"` uses a stronger leverage adjustment and is often considered in smaller samples.

``` python
fit_iid = pf.feols("Y ~ X1 + X2", data=data, vcov="iid")
fit_hc1 = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
fit_hc2 = pf.feols("Y ~ X1 + X2", data=data, vcov="HC2")
fit_hc3 = pf.feols("Y ~ X1 + X2", data=data, vcov="HC3")

pd.concat(
    {
        "iid": fit_iid.se(),
        "HC1": fit_hc1.se(),
        "HC2": fit_hc2.se(),
        "HC3": fit_hc3.se(),
    },
    axis=1,
)
```

|             | iid      | HC1      | HC2      | HC3      |
|-------------|----------|----------|----------|----------|
| Coefficient |          |          |          |          |
| Intercept   | 0.161614 | 0.156099 | 0.156851 | 0.157609 |
| X1          | 0.133160 | 0.128826 | 0.129614 | 0.130409 |
| X2          | 0.036854 | 0.037110 | 0.037424 | 0.037742 |

HC2 and HC3 are not supported for IV regressions or regressions with absorbed fixed effects. Use HC1, an appropriate clustered estimator, or another method justified by the design. Quantile regression accepts the HC1–HC3 labels but maps them to its `"hetero"` estimator; they are not distinct leverage corrections there.

## One-way and two-way clustering

Cluster-robust inference permits arbitrary error dependence within the named groups while relying on independence across groups. Cluster at the level where assignment or shared shocks induce dependence, not at whichever level produces the preferred p-value.

CRV1 is the usual cluster sandwich estimator. Put the estimator name in the dictionary key and the data column in its value:

``` python
fit_crv1 = pf.feols(
    "Y ~ X1 + X2 | f2",
    data=data,
    vcov={"CRV1": "f1"},
)
fit_crv1.tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|--------------|-----------|-----------|
| Coefficient |           |            |            |              |           |           |
| X1          | 1.314301  | 0.113233   | 11.607100  | 1.636680e-07 | 1.065078  | 1.563525  |
| X2          | -0.299524 | 0.027500   | -10.891717 | 3.128806e-07 | -0.360051 | -0.238997 |

For two-way clustering, join the two column names with `+`. PyFixest combines the one-way components using the standard inclusion-exclusion calculation.

``` python
fit_twoway = pf.feols(
    "Y ~ X1 + X2",
    data=data,
    vcov={"CRV1": "f1 + f2"},
)
fit_twoway.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| Intercept   | 2.535092  | 0.440336   | 5.757174  | 2.738119e-04 | 1.538983  | 3.531202  |
| X1          | 1.413638  | 0.105120   | 13.447800 | 2.901958e-07 | 1.175839  | 1.651437  |
| X2          | -0.276935 | 0.035648   | -7.768637 | 2.796177e-05 | -0.357576 | -0.196294 |

CRV3 is a leave-one-cluster-out jackknife. It is more computationally demanding, especially with absorbed fixed effects, because PyFixest may refit the model for each omitted cluster.

``` python
fit_crv3 = pf.feols(
    "Y ~ X1 + X2",
    data=data,
    vcov={"CRV3": "f1"},
)
fit_crv3.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| Intercept   | 2.535092  | 0.348917   | 7.265605  | 1.611780e-05 | 1.767131  | 3.303053  |
| X1          | 1.413638  | 0.143535   | 9.848739  | 8.607421e-07 | 1.097720  | 1.729556  |
| X2          | -0.276935 | 0.050369   | -5.498132 | 1.867614e-04 | -0.387796 | -0.166074 |

CRV3 is not supported for IV or quantile-regression models. Quantile regression supports one-way CRV1, not two-way clustering. A small number of clusters also makes asymptotic cluster inference fragile; where applicable, consider the wild cluster bootstrap below.

## Newey-West and Driscoll-Kraay HAC

Heteroskedasticity-and-autocorrelation-consistent (HAC) estimators address serial dependence. PyFixest provides Newey-West (`"NW"`) and Driscoll-Kraay (`"DK"`) covariance estimators.

Both require a numeric or date-like `time_id`. Panel Newey-West also takes a `panel_id`; without it, time values must be unique because the data are treated as one time series. Driscoll-Kraay requires both `time_id` and `panel_id`. Set the lag explicitly when it is part of the design; a time-series Newey-West call without a panel identifier currently requires it.

``` python
fit_nw = pf.feols(
    "y ~ x",
    data=panel,
    vcov="NW",
    vcov_kwargs={"time_id": "time_id", "panel_id": "panel_id", "lag": 2},
)

fit_dk = pf.feols(
    "y ~ x",
    data=panel,
    vcov="DK",
    vcov_kwargs={"time_id": "time_id", "panel_id": "panel_id", "lag": 2},
)

pd.concat({"Newey-West": fit_nw.se(), "Driscoll-Kraay": fit_dk.se()}, axis=1)
```

|             | Newey-West | Driscoll-Kraay |
|-------------|------------|----------------|
| Coefficient |            |                |
| Intercept   | 0.143183   | 0.082921       |
| x           | 0.129268   | 0.128941       |

HAC inference is available for linear and supported GLM/Poisson models, but not for quantile regression or frequency weights (`weights_type="fweights"`). The time spacing is currently treated as one between consecutive periods. Check that each panel-time pair is unique before fitting.

## Small-sample corrections

The `ssc=` argument controls finite-sample scaling and degrees of freedom. `pf.ssc()` exposes four choices:

- `k_adj` applies the coefficient-count adjustment.
- `k_fixef` controls how fixed-effect parameters enter that count.
- `G_adj` applies the cluster-count adjustment `G / (G - 1)`.
- `G_df` chooses how the cluster count is handled for multiway clustering.

``` python
fit_default_ssc = pf.feols(
    "Y ~ X1 | f2",
    data=data,
    vcov={"CRV1": "f1"},
)
fit_no_ssc = pf.feols(
    "Y ~ X1 | f2",
    data=data,
    vcov={"CRV1": "f1"},
    ssc=pf.ssc(k_adj=False, G_adj=False),
)

pd.concat(
    {"default SSC": fit_default_ssc.se(), "no SSC": fit_no_ssc.se()},
    axis=1,
)
```

|             | default SSC | no SSC   |
|-------------|-------------|----------|
| Coefficient |             |          |
| X1          | 0.135411    | 0.127438 |

The defaults intentionally follow PyFixest’s `fixest`-compatible behavior. Do not turn corrections off merely to reproduce another package: first match that package’s coefficient count, fixed-effect treatment, cluster adjustment, and degrees of freedom. See [Small-sample corrections](../explanation/ssc.md) for the formulas and compatibility details.

## Wild cluster bootstrap

When the number of clusters is small or their sizes are very uneven, the large-cluster approximation behind CRV1 can be poor. For an unweighted linear model, `wildboottest()` runs a heteroskedastic wild bootstrap or a one-way wild cluster bootstrap. Use a fixed seed for reproducibility.

``` python
fit_clustered = pf.feols(
    "Y ~ X1 + X2",
    data=data,
    vcov={"CRV1": "f1"},
)
fit_clustered.wildboottest(
    param="X1",
    cluster="f1",
    reps=99,
    seed=123,
)
```

    param                             X1
    t value           10.948616441288674
    Pr(>|t|)                         0.0
    bootstrap_type                    11
    inference                    CRV(f1)
    impose_null                     True
    ssc                          1.09833
    dtype: object

The wild bootstrap is not supported for IV, weighted, Poisson, or quantile regression models, and its clustered form accepts one cluster dimension only. The `wildboottest` dependency must be installed.

## Randomization inference

Randomization inference is design-based: it compares the observed statistic to statistics generated by the known assignment mechanism. `ritest()` permutes the named regressor and tests a zero null by default. Its `rng=` argument accepts a NumPy generator.

``` python
fit_ri = pf.feols("Y ~ X1 + X2", data=data, vcov="HC1")
fit_ri.ritest(
    resampvar="X1",
    reps=99,
    rng=np.random.default_rng(123),
)
```

    H0                                     X1=0
    ri-type                     randomization-c
    Estimate                 1.4136377875508426
    Pr(>|t|)                                0.0
    Std. Error (Pr(>|t|))                   0.0
    2.5% (Pr(>|t|))                         0.0
    97.5% (Pr(>|t|))                        0.0
    dtype: object

Use `cluster=` only when treatment was assigned at that cluster level and the assignment is constant within cluster. `type="randomization-c"` uses the coefficient; `type="randomization-t"` uses the t statistic and is slower. Randomization inference is not supported for IV or weighted models. It cannot repair an incorrectly specified assignment mechanism.

## Supported combinations

The most important restrictions are collected here. Estimator-specific pages remain authoritative for additional details.

| Method | Important restrictions |
|----|----|
| IID, HC1, CRV1 | Broadly supported; quantile regression uses its own kernel-based formulas and only one-way CRV1. |
| HC2, HC3 | No absorbed fixed effects and no IV; quantile-regression labels map to `"hetero"`. |
| CRV3 | Not IV or quantile regression; can be slow with fixed effects. |
| Two-way CRV1 | Not quantile regression. |
| NW, DK | Not quantile regression or frequency weights; require time metadata, and DK requires panel metadata. |
| Wild bootstrap | Not IV, WLS, Poisson, quantile regression, or multiway clustering. |
| Randomization inference | Not IV or WLS; the assignment mechanism must match the resampling scheme. |

Multiple-estimation results are held in `FixestMulti`. Set `vcov` and `ssc` in the estimation call when possible; use `FixestMulti.vcov()` or `FixestMulti.wildboottest()` only for combinations those methods support. Models fitted with `lean=True` or `store_data=False` may not retain the data and matrices required to recompute inference later.

## A reproducible inference workflow

1.  Write down which observations may share shocks and how treatment was assigned.
2.  Choose `vcov`, clustering columns, HAC identifiers, and SSC settings before inspecting significance.
3.  Report the estimator, cluster dimensions or lag, small-sample corrections, and the number of clusters or periods.
4.  Use bootstrap or randomization methods only when their resampling scheme matches the design.
5.  Treat unsupported combinations as errors, not invitations to silently choose a different covariance estimator.
