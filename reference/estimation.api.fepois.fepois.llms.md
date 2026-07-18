# estimation.api.fepois.fepois

``` python
estimation.api.fepois.fepois(
    fml,
    data,
    vcov=None,
    vcov_kwargs=None,
    weights=None,
    weights_type='aweights',
    offset=None,
    ssc=None,
    fixef_rm='singleton',
    iwls_tol=1e-08,
    iwls_maxiter=25,
    collin_tol=1e-09,
    separation_check=None,
    solver='scipy.linalg.solve',
    demeaner=None,
    drop_intercept=False,
    copy_data=True,
    store_data=True,
    lean=False,
    context=None,
    split=None,
    fsplit=None,
)
```

Estimate Poisson regression model with fixed effects using the `ppmlhdfe` algorithm.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| fml | str | A two-sided formula string using fixest formula syntax. Syntax: “Y ~ X1 + X2 \| FE1 + FE2”. “\|” separates left-hand side and fixed effects. Special syntax includes: - Stepwise regressions (sw, sw0) - Cumulative stepwise regression (csw, csw0) - Multiple dependent variables (Y1 + Y2 ~ X) - Interaction of variables (i(X1,X2)) - Interacted fixed effects (fe1^fe2) Compatible with formula parsing via the formulaic module. | *required* |
| data | DataFrameType | A pandas or polars dataframe containing the variables in the formula. | *required* |
| vcov | Union\[VcovTypeOptions, dict\[str, str\]\] | Type of variance-covariance matrix for inference. Options include “iid”, “hetero”, “HC1”, “HC2”, “HC3”, “NW” for Newey-West HAC standard errors, “DK” for Driscoll-Kraay HAC standard errors, or a dictionary for CRV1/CRV3 inference. Note that NW and DK require to pass additional keyword arguments via the `vcov_kwargs` argument. For time-series HAC, you need to pass the ‘time_id’ column. For panel-HAC, you need to add pass both ‘time_id’ and ‘panel_id’. See `vcov_kwargs` for details. | `None` |
| vcov_kwargs | Optional\[dict\[str, any\]\] | Additional keyword arguments to pass to the vcov function. These keywoards include “lag” for the number of lag to use in the Newey-West (NW) and Driscoll-Kraay (DK) HAC standard errors. “time_id” for the time ID used for NW and DK standard errors, and “panel_id” for the panel identifier used for NW and DK standard errors. Currently, the the time difference between consecutive time periods is always treated as 1. More flexible time-step selection is work in progress. | `None` |
| weights | Union\[None, str\], optional. | Default is None. Weights for weighted Poisson regression. If None, all observations are weighted equally. If a string, the name of the column in `data` that contains the weights. | `None` |
| weights_type | WeightsTypeOptions | Options include `aweights` or `fweights`. `aweights` implement analytic or precision weights, while `fweights` implement frequency weights. Frequency weights are useful for compressed count data where identical observations are aggregated. For details see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/. | `'aweights'` |
| offset | str | Default is None. Name of a numeric column in `data` to use as an offset in the Poisson regression. The offset is added to the linear predictor with its coefficient fixed at 1. This is useful for modeling rates when exposure differs across observations; pass the exposure on the log scale, e.g. `offset="log_population"`. | `None` |
| ssc | str | A ssc object specifying the small sample correction for inference. | `None` |
| fixef_rm | FixedRmOptions | Specifies whether to drop singleton fixed effects. Can be equal to “singletons” (default) or “none”. “singletons” will drop singleton fixed effects. This will not impact point estimates but it will impact standard errors. | `'singleton'` |
| iwls_tol | Optional\[float\] | Tolerance for IWLS convergence, by default 1e-08. | `1e-08` |
| iwls_maxiter | Optional\[float\] | Maximum number of iterations for IWLS convergence, by default 25. | `25` |
| collin_tol | float | Tolerance for collinearity check, by default 1e-10. | `1e-09` |
| separation_check | list\[str\] \| None | Methods to identify and drop separated observations. Either “fe” or “ir”. Executes “fe” by default (when None). | `None` |
| solver | SolverOptions, optional. | The solver to use for the regression. Can be “np.linalg.lstsq”, “np.linalg.solve”, “scipy.linalg.solve” and “scipy.sparse.linalg.lsqr”. Defaults to “scipy.linalg.solve”. | `'scipy.linalg.solve'` |
| demeaner | AnyDemeaner \| None | Typed demeaner configuration. Controls the fixed-effects demeaning backend, tolerance, and iteration limits. Accepts a `MapDemeaner` or `LsmrDemeaner` instance. Defaults to `MapDemeaner()` (Rust MAP algorithm, tol=1e-6, maxiter=10_000). For other options - including the optional Numba backend and the torch-based LSMR backends - see the [Demeaner Backends vignette](../../how-to/demeaner-backends.qmd). .. deprecated:: The `cupy` / `scipy` LSMR backends are deprecated and will be removed in a future release. Replacements: - cupy LSMR on GPU → `LsmrDemeaner(backend="torch", device="cuda")`. - Scipy / cupy LSMR on CPU → `LsmrDemeaner()` (the default within backend). | `None` |
| drop_intercept | bool | Whether to drop the intercept from the model, by default False. | `False` |
| copy_data | bool | Whether to copy the data before estimation, by default True. If set to False, the data is not copied, which can save memory but may lead to unintended changes in the input data outside of `fepois`. For example, the input data set is re-index within the function. As far as I know, the only other relevant case is when using interacted fixed effects, in which case you’ll find a column with interacted fixed effects in the data set. | `True` |
| store_data | bool | Whether to store the data in the model object, by default True. If set to False, the data is not stored in the model object, which can improve performance and save memory. However, it will no longer be possible to access the data via the `data` attribute of the model object. This has impact on post-estimation capabilities that rely on the data, e.g. `predict()` or `vcov()`. | `True` |
| lean | bool | False by default. If True, then all large objects are removed from the returned result: this will save memory but will block the possibility to use many methods. It is recommended to use the argument vcov to obtain the appropriate standard-errors at estimation time, since obtaining different SEs won’t be possible afterwards. | `False` |
| context | int or Mapping\[str, Any\] | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. | `None` |
| split | str \| None | A character string, i.e. ‘split = var’. If provided, the sample is split according to the variable and one estimation is performed for each value of that variable. If you also want to include the estimation for the full sample, use the argument fsplit instead. | `None` |
| fsplit | str \| None | This argument is the same as split but also includes the full sample as the first estimation. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | object | An instance of the [Fepois](../reference/estimation.models.fepois_.Fepois.llms.md) class or an instance of class [FixestMulti](../reference/estimation.FixestMulti_.FixestMulti.llms.md) for multiple models specified via `fml`. |

## Examples

The `fepois()` function estimates Poisson models with the same formula interface as `feols()`. Fixed effects are specified after the `|` symbol.

``` python
import pyfixest as pf

data = pf.get_data(model="Fepois")
fit = pf.fepois("Y ~ X1 + X2 | f1 + f2", data)
fit.summary()
```

    ###

    Estimation:  Poisson
    Dep. var.: Y, Fixed effects: f1 + f2
    sample: None = all
    Inference:  iid
    Observations:  995

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.007 |        0.042 |    -0.157 |      0.875 | -0.089 |   0.076 |
    | X2            |     -0.015 |        0.011 |    -1.317 |      0.188 | -0.037 |   0.007 |
    ---
    Deviance: 1068.169 

Cluster-robust inference uses the same `vcov` syntax as `feols()`:

``` python
fit_crv = pf.fepois("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
fit_crv.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | -0.006591 | 0.035301   | -0.186711 | 0.851887    | -0.075780 | 0.062598 |
| X2          | -0.014924 | 0.010467   | -1.425778 | 0.153932    | -0.035439 | 0.005591 |

To model rates, keep the dependent variable as a count and pass `log(exposure)` as the `offset`. For example, with population as the exposure:

``` python
import numpy as np

data["population"] = np.random.default_rng(123).integers(
    50_000, 500_000, size=len(data)
)
data["log_population"] = np.log(data["population"])

fit_rate = pf.fepois(
    "Y ~ X1 + X2 | f1 + f2",
    data=data,
    offset="log_population",
)
fit_rate.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | -0.003015 | 0.042016   | -0.071767 | 0.942787    | -0.085366 | 0.079335 |
| X2          | -0.018169 | 0.011338   | -1.602503 | 0.109044    | -0.040391 | 0.004053 |

Multiple-estimation and sample-splitting features also work as in `feols()`:

``` python
fits = pf.fepois("Y ~ X1 | sw0(f1, f2)", data)
pf.etable(fits)
```

[TABLE]

Shared arguments such as `vcov`, `ssc`, `split`, `fsplit`, `context`, and typed demeaners are documented in the [feols() reference](../reference/estimation.api.feols.feols.llms.md). For applied examples, see the [Poisson & GLMs tutorial](../tutorials/poisson-glm.llms.md).
