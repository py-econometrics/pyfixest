<!-- Generated from docs/reference/estimation.api.quantreg.quantreg.qmd; do not edit. -->

# estimation.api.quantreg.quantreg

```python
estimation.api.quantreg.quantreg(
    fml,
    data,
    vcov='nid',
    quantile=0.5,
    method='fn',
    multi_method='cfm1',
    tol=1e-06,
    maxiter=None,
    ssc=None,
    collin_tol=1e-09,
    separation_check=None,
    drop_intercept=False,
    copy_data=True,
    store_data=True,
    lean=False,
    context=None,
    split=None,
    fsplit=None,
    seed=None,
)
```

Fit a quantile regression model using the interior point algorithm from Portnoy and Koenker (1997).
Note that the interior point algorithm assumes independent observations.

## Parameters

| Name             | Type                                 | Description                                                                                                                                                                                                                                                                                                                            | Default    |
|------------------|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| fml              | str                                  | A two-sided formula string using fixest formula syntax. In contrast to `feols()` and `feglm()`, no fixed effects formula syntax is supported.                                                                                                                                                                                          | _required_ |
| data             | DataFrameType                        | A pandas or polars dataframe containing the variables in the formula.                                                                                                                                                                                                                                                                  | _required_ |
| vcov             | QuantregVcovType or dict\[str, str\] | Variance-covariance estimator. Defaults to `"nid"`. String options are `"iid"`, `"nid"`, `"hetero"`, `"HC1"`, `"HC2"`, and `"HC3"`; HC1-HC3 are aliases for `"hetero"`. One-way clustering uses a dictionary such as `{"CRV1": "cluster"}`. HAC estimators are not supported.                                                          | `'nid'`    |
| quantile         | float or list\[float\]               | Quantile or quantiles to estimate, strictly between zero and one. A list returns a `FixestMulti` containing one `Quantreg` result per quantile. Defaults to `0.5`.                                                                                                                                                                     | `0.5`      |
| method           | QuantregMethodOptions                | Fitting algorithm. `"fn"` implements the Frisch-Newton interior-point algorithm described in Portnoy and Koenker (1997). `"pfn"` implements a variant with preprocessing steps from the same paper. The preprocessing can accelerate large samples and uses `seed` for its random number generator.                                    | `'fn'`     |
| multi_method     | QuantregMultiOptions                 | Algorithm for a list of quantiles. `"cfm1"` (default) implements algorithm 2 and `"cfm2"` implements algorithm 3 from Chernozhukov, Fernández-Val, and Melly (2019).                                                                                                                                                                   | `'cfm1'`   |
| tol              | float                                | The tolerance for the algorithm. Defaults to 1e-06. As in R's quantreg package, the algorithm stops when the relative change in the duality gap is less than tol.                                                                                                                                                                      | `1e-06`    |
| maxiter          | int                                  | The maximum number of iterations. If None, maxiter = the number of observations in the model (as in R's quantreg package via nit(3) = n).                                                                                                                                                                                              | `None`     |
| ssc              | SscConfig                            | Small-sample correction created by `ssc()`. `None` uses `ssc(k_adj=True, k_fixef="nonnested", G_adj=True, G_df="min")`. To match software that applies no small-sample correction, use `ssc(k_adj=False, G_adj=False)`.                                                                                                                | `None`     |
| collin_tol       | float                                | Tolerance for the collinearity check. Defaults to `1e-9`.                                                                                                                                                                                                                                                                              | `1e-09`    |
| separation_check | list\[str\]                          | Methods to identify and drop separated observations. Not used in quantile regression.                                                                                                                                                                                                                                                  | `None`     |
| drop_intercept   | bool                                 | Whether to drop the intercept from the model, by default False.                                                                                                                                                                                                                                                                        | `False`    |
| copy_data        | bool                                 | Whether to copy the data before estimation, by default True. If set to False, the data is not copied, which can save memory but may lead to unintended changes in the input data outside of `quantreg`.                                                                                                                                | `True`     |
| store_data       | bool                                 | Whether to store the data in the model object, by default True. If set to False, the data is not stored in the model object, which can improve performance and save memory. However, it will no longer be possible to access the data via the `data` attribute of the model object.                                                    | `True`     |
| lean             | bool                                 | False by default. If True, then all large objects are removed from the returned result: this will save memory but will block the possibility to use many methods. It is recommended to use the argument vcov to obtain the appropriate standard-errors at estimation time, since obtaining different SEs won't be possible afterwards. | `False`    |
| context          | int or Mapping\[str, Any\]           | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment.                                                                   | `None`     |
| split            | str                                  | A character string, i.e. 'split = var'. If provided, the sample is split according to the variable and one estimation is performed for each value of that variable. If you also want to include the estimation for the full sample, use the argument fsplit instead.                                                                   | `None`     |
| fsplit           | str                                  | This argument is the same as split but also includes the full sample as the first estimation.                                                                                                                                                                                                                                          | `None`     |
| seed             | int \| None                          | A random seed for reproducibility. If None, no seed is set. Only relevant for the "pfn" method. The "fn" method is deterministic and does not require a seed.                                                                                                                                                                          | `None`     |

## Returns

| Name   | Type                    | Description                                                                                                                                                                                                         |
|--------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | Quantreg or FixestMulti | A [Quantreg](estimation.quantreg.quantreg_.Quantreg.md), or [FixestMulti](estimation.FixestMulti_.FixestMulti.md) when quantiles, formula syntax, or split options produce multiple models. |

## Examples

The following example regresses `Y` on `X1` and `X2` at the median (0.5 quantile):

```python
import pyfixest as pf

data = pf.get_data()

fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
fit.summary()
```

To fit multiple quantiles in one call:

```python
fits = pf.quantreg("Y ~ X1 + X2", data, quantile=[0.1, 0.5, 0.9])
pf.qplot(fits)
```

Arguments such as `split`, `fsplit`, `context`, `lean`, and `copy_data`
behave as in `feols()`, but quantile regression does not support fixed-effects
formula syntax. For details around inference, fast fitting, and visualization
of the full quantile regression process, see the
[quantile regression tutorial](../tutorials/quantile-regression.md).
