# Quantreg

``` python
Quantreg(
    FixestFormula,
    data,
    *,
    options,
    lookup_demeaned_data,
    sample_split=None,
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

| Name | Type | Description |
|----|----|----|
| solution | QuantregSolution | Output of the interior point solver: coefficients, convergence flag, number of iterations, and final primal and dual iterates. With `multi_method="cfm2"`, the solver runs only for the central quantile; the other quantiles get a single Newton step and have no solution. Dropped by `lean=True`. |

## Methods

| Name | Description |
|----|----|
| [Quantreg.drop_multicol_vars](#pyfixest.estimation.quantreg.quantreg_.Quantreg.drop_multicol_vars) | Remove collinear regressors using the same rank check as OLS. |
| [Quantreg.fit_qreg_fn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_fn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver. |
| [Quantreg.fit_qreg_pfn](#pyfixest.estimation.quantreg.quantreg_.Quantreg.fit_qreg_pfn) | Fit a quantile regression model using the Frisch-Newton Interior Point Solver with pre-processing. |
| [Quantreg.get_fit](#pyfixest.estimation.quantreg.quantreg_.Quantreg.get_fit) | Fit a quantile regression model using the interior point method. |
| [Quantreg.prepare_model_matrix](#pyfixest.estimation.quantreg.quantreg_.Quantreg.prepare_model_matrix) | Prepare model inputs for estimation. |
| [Quantreg.to_array](#pyfixest.estimation.quantreg.quantreg_.Quantreg.to_array) | Publish quantile-regression arrays from the formula state. |

### Quantreg.drop_multicol_vars

``` python
drop_multicol_vars()
```

Remove collinear regressors using the same rank check as OLS.

Quantile models do not support fixed effects, so within_data holds the original response and regressor arrays without demeaning.

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

### Quantreg.prepare_model_matrix

``` python
prepare_model_matrix()
```

Prepare model inputs for estimation.

### Quantreg.to_array

``` python
to_array()
```

Publish quantile-regression arrays from the formula state.
