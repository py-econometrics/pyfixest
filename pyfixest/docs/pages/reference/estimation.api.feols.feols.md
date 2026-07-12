<!-- Generated from docs/reference/estimation.api.feols.feols.qmd; do not edit. -->

# estimation.api.feols.feols

```python
estimation.api.feols.feols(
    fml,
    data,
    vcov=None,
    vcov_kwargs=None,
    weights=None,
    ssc=None,
    fixef_rm='singleton',
    collin_tol=1e-09,
    drop_intercept=False,
    copy_data=True,
    store_data=True,
    lean=False,
    weights_type='aweights',
    solver='scipy.linalg.solve',
    demeaner=None,
    use_compression=False,
    reps=100,
    context=None,
    seed=None,
    split=None,
    fsplit=None,
)
```

Estimate a linear regression model with fixed effects using fixest formula syntax.

Returns an object of type [Feols](estimation.models.feols_.Feols.md) or
[Feiv](estimation.models.feiv_.Feiv.md) (when using instrumental variables).

## Parameters

| Name            | Type                                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Default                |
|-----------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| fml             | str                                    | A three-sided formula string using fixest formula syntax. Syntax: "Y ~ X1 + X2 \| FE1 + FE2 \| X1 ~ Z1". "\|" separates dependent variable, fixed effects, and instruments. Special syntax includes stepwise regressions, cumulative stepwise regression, multiple dependent variables, interaction of variables (i(X1,X2)), and interacted fixed effects (fe1^fe2).                                                                                                                                                                                                                                                                                                                                                                | _required_             |
| data            | DataFrameType                          | A pandas or polars dataframe containing the variables in the formula.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | _required_             |
| vcov            | RegressionVcovType or dict\[str, str\] | Variance-covariance estimator. `None` defaults to `"iid"`. String options are `"iid"`, `"hetero"`, `"HC1"`, `"HC2"`, `"HC3"`, `"NW"`, and `"DK"`. Clustered inference uses `{"CRV1": "cluster"}` or `{"CRV3": "cluster"}`; join column names with `+` for multiway CRV1. CRV3 is not supported for IV models. NW and DK require `vcov_kwargs`.                                                                                                                                                                                                                                                                                                                                                                                      | `None`                 |
| vcov_kwargs     | VcovKwargs                             | HAC configuration. Pass `lag` and `time_id` for time-series NW; add `panel_id` for panel NW or DK. Consecutive time values are currently treated as one period apart.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `None`                 |
| weights         | Union\[None, str\], optional.          | Default is None. Weights for WLS estimation. If None, all observations are weighted equally. If a string, the name of the column in `data` that contains the weights.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `None`                 |
| ssc             | SscConfig                              | Small-sample correction created by `ssc()`. `None` uses `ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `None`                 |
| fixef_rm        | FixedRmOptions                         | Specifies whether to drop singleton fixed effects. Can be equal to `"singleton"` (default) or `"none"`. `"singleton"` drops singleton fixed effects. This does not affect point estimates but it will impact standard errors.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `'singleton'`          |
| collin_tol      | float                                  | Tolerance for the collinearity check. Defaults to `1e-9`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `1e-09`                |
| drop_intercept  | bool                                   | Whether to drop the intercept from the model, by default False.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `False`                |
| copy_data       | bool                                   | Whether to copy the data before estimation, by default True. If set to False, the data is not copied, which can save memory but may lead to unintended changes in the input data outside of `feols`. For example, the input data set is re-index within the function. As far as I know, the only other relevant case is when using interacted fixed effects, in which case you'll find a column with interacted fixed effects in the data set.                                                                                                                                                                                                                                                                                      | `True`                 |
| store_data      | bool                                   | Whether to store the data in the model object, by default True. If set to False, the data is not stored in the model object, which can improve performance and save memory. However, it will no longer be possible to access the data via the `data` attribute of the model object. This has impact on post-estimation capabilities that rely on the data, e.g. `predict()` or `vcov()`.                                                                                                                                                                                                                                                                                                                                            | `True`                 |
| lean            | bool                                   | False by default. If True, then all large objects are removed from the returned result: this will save memory but will block the possibility to use many methods. It is recommended to use the argument vcov to obtain the appropriate standard-errors at estimation time, since obtaining different SEs won't be possible afterwards.                                                                                                                                                                                                                                                                                                                                                                                              | `False`                |
| weights_type    | WeightsType                            | Options include `aweights` or `fweights`. `aweights` implement analytic or precision weights, while `fweights` implement frequency weights. For details see this blog post: https://notstatschat.rbind.io/2020/08/04/weights-in-statistics/.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `'aweights'`           |
| solver          | SolverOptions, optional.               | The solver to use for the regression. Can be "np.linalg.lstsq", "np.linalg.solve", "scipy.linalg.solve" and "scipy.sparse.linalg.lsqr". Defaults to "scipy.linalg.solve".                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `'scipy.linalg.solve'` |
| demeaner        | AnyDemeaner \| None                    | Typed demeaner configuration. Controls the fixed-effects demeaning backend, tolerance, and iteration limits. Accepts a `MapDemeaner` or `LsmrDemeaner` instance. Defaults to `MapDemeaner()` (Rust MAP algorithm, tol=1e-6, maxiter=10_000). For other options - including the optional Numba backend and the torch-based LSMR backends - see the [Demeaner Backends vignette](../how-to/demeaner-backends.md).  .. deprecated::     The ``cupy`` / ``scipy`` LSMR backends are deprecated and will     be removed in a future release. Replacements:      - cupy LSMR on GPU →       ``LsmrDemeaner(backend="torch", device="cuda")``.     - Scipy / cupy LSMR on CPU → ``LsmrDemeaner()``       (the default within backend). | `None`                 |
| use_compression | bool                                   | .. deprecated::     ``use_compression`` is deprecated and no longer supported. Passing     ``use_compression=True`` raises a ``NotImplementedError``. For     out-of-memory regression on large datasets, consider using the     `duckreg <https://github.com/py-econometrics/duckreg>`_ package instead.                                                                                                                                                                                                                                                                                                                                                                                                                           | `False`                |
| reps            | int                                    | Deprecated legacy argument for compressed regression bootstrap inference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `100`                  |
| context         | int or Mapping\[str, Any\]             | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment.                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `None`                 |
| seed            | int \| None                            | Deprecated legacy argument for compressed regression bootstrap inference.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `None`                 |
| split           | str \| None                            | A character string, i.e. 'split = var'. If provided, the sample is split according to the variable and one estimation is performed for each value of that variable. If you also want to include the estimation for the full sample, use the argument fsplit instead.                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `None`                 |
| fsplit          | str \| None                            | This argument is the same as split but also includes the full sample as the first estimation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `None`                 |

## Returns

| Name   | Type                 | Description                                                                                                                                                                                                                                                          |
|--------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | Feols or FixestMulti | A [Feols](estimation.models.feols_.Feols.md), its [Feiv](estimation.models.feiv_.Feiv.md) subclass for IV, or [FixestMulti](estimation.FixestMulti_.FixestMulti.md) when the formula or split options expand to multiple models. |

## Examples

As in `fixest`, the [feols()](estimation.api.feols.feols.md) function can be used to
estimate a simple linear regression model with fixed effects.
The following example regresses `Y` on `X1` and `X2` with fixed effects for
`f1` and `f2`: fixed effects are specified after the `|` symbol.

```python
import pyfixest as pf
import pandas as pd
import numpy as np

data = pf.get_data()

fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
fit.summary()
```

Calling `feols()` returns an instance of the [Feols](estimation.models.feols_.Feols.md)
class. The `summary()` method can be used to print the results.

An alternative way to retrieve model results is via the `tidy()` method, which
returns a pandas dataframe with the estimated coefficients, standard errors,
t-statistics, and p-values.

```python
fit.tidy()
```

You can also access common outputs via dedicated methods,
e.g. `fit.coef()` for the coefficients, `fit.se()` for the standard errors,
`fit.tstat()` for the t-statistics, `fit.pvalue()` for the p-values, and
`fit.confint()` for the confidence intervals.

The employed type of inference can be specified via the `vcov` argument. For compatibility
with `fixest`, if vcov is not provided, `PyFixest` always employs "iid" inference by default
starting with pyfixest 0.31.0. Prior to pyfixest 0.31.0, if vcov was not provided, `PyFixest`
would cluster by the first fixed effect if no vcov was provided.

```python
fit1 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov="iid")
fit2 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov="hetero")
fit3 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1"})
```

Supported inference types are "iid", "hetero", "HC1", "HC2", "HC3", and
"CRV1"/"CRV3". Clustered standard errors are specified via a dictionary,
e.g. `{"CRV1": "f1"}` for CRV1 inference with clustering by `f1` or
`{"CRV3": "f1"}` for CRV3 inference with clustering by `f1`. For two-way
clustering, you can provide a formula string, e.g. `{"CRV1": "f1 + f2"}` for
CRV1 inference with clustering by `f1`.

```python
fit4 = pf.feols("Y ~ X1 + X2 | f1 + f2", data, vcov={"CRV1": "f1 + f2"})
```

Inference can be adjusted post estimation via the `vcov` method:

```python
fit.summary()
fit.vcov("iid").summary()
```

The `ssc` argument controls small-sample corrections. The default is
`pf.ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")`.

`feols()` supports a range of multiple estimation syntax, i.e. you can estimate
multiple models in one call. The following example estimates two models, one with
fixed effects for `f1` and one with fixed effects for `f2` using the `sw()` syntax.

```python
fit = pf.feols("Y ~ X1 + X2 | sw(f1, f2)", data)
type(fit)
```

The returned object is an instance of the `FixestMulti` class. You can access
the results of the first model via `fit.fetch_model(0)` and the results of
the second model via `fit.fetch_model(1)`. You can compare the model results
via the `etable()` function:

```python
pf.etable(fit)
```

Other supported multiple estimation syntax include `sw0()`, `csw()` and `csw0()`.
While `sw()` adds variables in a "stepwise" fashion, `csw()` does so cumulatively.

```python
fit = pf.feols("Y ~ X1 + X2 | csw(f1, f2)", data)
pf.etable(fit)
```

The `sw0()` and `csw0()` syntax are similar to `sw()` and `csw()`, but start
with a model that excludes the variables specified in `sw()` and `csw()`:

```python
fit = pf.feols("Y ~ X1 + X2 | sw0(f1, f2)", data)
pf.etable(fit)
```

The `feols()` function also supports multiple dependent variables. The following
example estimates two models, one with `Y1` as the dependent variable and one
with `Y2` as the dependent variable.

```python
fit = pf.feols("Y + Y2 ~ X1 | f1 + f2", data)
pf.etable(fit)
```

It is possible to combine different multiple estimation operators:

```python
fit = pf.feols("Y + Y2 ~ X1 | sw(f1, f2)", data)
pf.etable(fit)
```

Multiple estimation can reuse demeaning work. `feols()` records the call in an
`EstimationConfig`; `parse_formula()` expands it into model specifications, and
`runner.run_estimation()` supplies each model with a shared cache for compatible
specifications. `FixestMulti` only stores the fitted results.

Additionally, you can fit models on different samples via the split and fsplit
arguments. The split argument splits the sample according to the variable
specified in the argument, while the fsplit argument also includes the full
sample in the estimation.

```python
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data, split = "f1")
pf.etable(fit)
```

Besides OLS, `feols()` also supports IV estimation via three-part formulas.
IV models return an instance of the [Feiv](estimation.models.feiv_.Feiv.md)
class (which inherits from [Feols](estimation.models.feols_.Feols.md)).

```python
fit_iv = pf.feols("Y ~ X2 | f1 + f2 | X1 ~ Z1", data)
type(fit_iv)
```

Here, `X1` is the endogenous variable and `Z1` is the instrument. `f1` and `f2`
are the fixed effects, as before. To estimate IV models without fixed effects,
simply omit the fixed effects part of the formula:

```python
fit_iv2 = pf.feols("Y ~ X2 | X1 ~ Z1", data)
fit_iv2.tidy()
```

You can compare OLS and IV estimates side by side via `etable()`:

```python
fit_ols = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
pf.etable([fit_ols, fit_iv])
```

To diagnose weak instruments, use the `IV_Diag()` method, which computes
the first-stage F-statistic and the effective F-statistic
(Olea and Pflueger, 2013):

```python
fit_iv.IV_Diag()
print("First-stage F-statistic:", round(fit_iv.first_stage_f_statistic, 3))
print("Effective F-statistic:", round(fit_iv.effective_f_statistic, 3))
```

You can also access the first-stage regression as a `Feols` object via
`first_stage_model` and display both stages with `etable()`:

```python
first_stage = fit_iv.first_stage_model
pf.etable([first_stage, fit_iv])
```

Last, `feols()` supports interaction of variables via the `i()` syntax.
For a compact overview of formula features including `i()`, see the
[formula syntax tutorial](../tutorials/formula-syntax.md).

You can pass custom transforms via the `context` argument. If you set `context = 0`, all
functions from the level of the call to `feols()` will be available:

```python
def _lspline(series: pd.Series, knots: list[float]) -> np.array:
    'Generate a linear spline design matrix for the input series based on knots.'
    vector = series.values
    columns = []

    for i, knot in enumerate(knots):
        column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
        columns.append(column)
        vector = vector - column

    # Add the remainder as the last column
    columns.append(vector)

    # Combine columns into a design matrix
    return np.column_stack(columns)

spline_split = _lspline(data["X2"], [0, 1])
data["X2_0"] = spline_split[:, 0]
data["0_X2_1"] = spline_split[:, 1]
data["1_X2"] = spline_split[:, 2]

explicit_fit = pf.feols("Y ~ X2_0 + 0_X2_1 + 1_X2 | f1 + f2", data=data)
# set context = 0 to make _lspline available for feols' internal call to Formulaic.model_matrix
context_captured_fit = pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=data, context = 0)
# or provide it as a dict / mapping
context_captured_fit_map = pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=data, context = {"_lspline":_lspline})

pf.etable([explicit_fit, context_captured_fit, context_captured_fit_map])
```

After fitting a model via `feols()`, you can use the `predict()` method to
get the predicted values:

```python
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
fit.predict()[0:5]
```

The `predict()` method also supports a `newdata` argument to predict on new data,
which returns a numpy array of the predicted values:

```python
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
fit.predict(newdata=data)[0:5]
```

Last, you can plot the results of a model via the `coefplot()` method:

```python
fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
fit.coefplot()
```

We can conduct a regression decomposition via the `decompose()` method, which implements
a regression decomposition following the method developed in Gelbach (2016):

```python
import re
import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data

data_gelbach = gelbach_data(nobs = 1000)
fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data_gelbach)

# simple decomposition
res = fit.decompose(param = "x1")
res.etable()

# group covariates via "combine_covariates" argument
res = fit.decompose(param = "x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]})
res.etable()

# group covariates via regex
res = fit.decompose(param="x1", combine_covariates={"g1": re.compile("x2[1-2]"), "g2": re.compile("x23")})
```

Objects of type `Feols` support a range of other methods to conduct inference.
For example, you can run a wild (cluster) bootstrap via the `wildboottest()` method:

```python
fit = pf.feols("Y ~ X1 + X2", data)
fit.wildboottest(param = "X1", reps=1000)
```
would run a wild bootstrap test for the coefficient of `X1` with 1000
bootstrap repetitions.

For a wild cluster bootstrap, you can specify the cluster variable
  via the `cluster` argument:

```python
fit.wildboottest(param = "X1", reps=1000, cluster="group_id")
```

The `ritest()` method can be used to conduct randomization inference:

```python
fit.ritest(resampvar = "X1", reps=1000)
```

Last, you can compute the cluster causal variance estimator by Athey et
al by using the `ccv()` method:

```python
import numpy as np
rng = np.random.default_rng(1234)
data["D"] = rng.choice([0, 1], size = data.shape[0])
fit_D = pf.feols("Y ~ D", data = data)
fit_D.ccv(treatment = "D", cluster = "group_id")
```
