# Quantreg

``` python
Quantreg(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    lookup_demeaned_data,
    solver='np.linalg.solve',
    demeaner=None,
    store_data=True,
    copy_data=True,
    lean=False,
    context=0,
    sample_split_var=None,
    sample_split_value=None,
    quantile=0.5,
    method='fn',
    quantile_tol=1e-06,
    quantile_maxiter=None,
    seed=None,
)
```

Quantile regression model.

Returned by [quantreg()](../reference/estimation.api.quantreg.quantreg.llms.md). Fits the conditional quantile of the outcome instead of the conditional mean, which allows the effect of a covariate to differ across the outcome distribution. Estimated via the interior point algorithm of Portnoy and Koenker (1997), [Statistical Science](https://doi.org/10.1214/ss/1030037960).

## Examples

``` python
import pyfixest as pf

data = pf.get_data()

fit = pf.quantreg("Y ~ X1 + X2", data, quantile=0.5)
fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| Intercept   | 0.997881  | 0.172038   | 5.800355  | 8.884297e-09 | 0.660282  | 1.335480  |
| X1          | -1.070846 | 0.122024   | -8.775730 | 0.000000e+00 | -1.310299 | -0.831393 |
| X2          | -0.182325 | 0.031911   | -5.713491 | 1.461492e-08 | -0.244946 | -0.119704 |

Several quantiles can be estimated in one call. [qplot()](../reference/report.qplot.llms.md) plots the resulting coefficients.

``` python
fits = pf.quantreg("Y ~ X1 + X2", data, quantile=[0.25, 0.5, 0.75])
pf.etable(fits)
```

[TABLE]

See the [quantile regression tutorial](../tutorials/quantile-regression.llms.md) for details.

## Attributes

| Name | Description |
|----|----|
| [Quantreg.objective_value](#pyfixest.estimation.quantreg.quantreg_.Quantreg.objective_value) | Compute the total loss of the quantile regression model. |

## Methods

| Name | Description |
|----|----|
| [Quantreg.fit_qreg_fn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_fn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver. |
| [Quantreg.fit_qreg_pfn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_pfn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing. |
| [Quantreg.get_fit](#pyfixest.estimation.quantreg.quantreg_.Quantreg.get_fit) | Fit a quantile regression model using the interior point method. |
| [Quantreg.get_performance](#pyfixest.estimation.quantreg.quantreg_.Quantreg.get_performance) | Compute performance metrics for the quantile regression model. |
| [Quantreg.prepare_model_matrix](#pyfixest.estimation.quantreg.quantreg_.Quantreg.prepare_model_matrix) | Prepare model inputs for estimation. |
| [Quantreg.to_array](#pyfixest.estimation.quantreg.quantreg_.Quantreg.to_array) | Turn estimation DataFrames to np arrays. |

### Quantreg.fit_qreg_fn

``` python
fit_qreg_fn(X, Y, q, tol=None, maxiter=None, beta_init=None)
```

Fit a quantile regression model using the Frisch-Newton Interior Point Solver.

### Quantreg.fit_qreg_pfn

``` python
fit_qreg_pfn(
    X,
    Y,
    q,
    m=None,
    tol=None,
    maxiter=None,
    beta_init=None,
    rng=None,
    eta=None,
)
```

Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing.

### Quantreg.get_fit

``` python
get_fit()
```

Fit a quantile regression model using the interior point method.

### Quantreg.get_performance

``` python
get_performance()
```

Compute performance metrics for the quantile regression model.

### Quantreg.prepare_model_matrix

``` python
prepare_model_matrix()
```

Prepare model inputs for estimation.

### Quantreg.to_array

``` python
to_array()
```

Turn estimation DataFrames to np arrays.
