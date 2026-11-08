# CoefficientTable

``` python
CoefficientTable(estimate, se, tstat, pvalue, conf_int, alpha)
```

Coefficient estimates with their standard errors, statistics, and bounds.

`get_inference()` publishes one value as `fit.coeftable` from the current `fit.variance_covariance`; it is the table that fixest’s `coeftable()` prints, plus the confidence bounds of `confint()`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| estimate | NDArray\[np.float64\] | Coefficient estimates, shape (n_coefficients,). | *required* |
| se | NDArray\[np.float64\] | Standard errors, the square root of the covariance diagonal. | *required* |
| tstat | NDArray\[np.float64\] | `estimate / se`. | *required* |
| pvalue | NDArray\[np.float64\] | Two-sided p-values from the model’s reference distribution with `variance_covariance.df_t` degrees of freedom. | *required* |
| conf_int | NDArray\[np.float64\] | Lower and upper confidence bounds, shape (2, n_coefficients), at level `1 - alpha`. | *required* |
| alpha | float | Significance level of the bounds. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data(), vcov="HC1")
table = fit.coeftable
table.estimate, table.se, table.conf_int
```

    (array([-0.94944106]),
     array([0.06736535]),
     array([[-1.08164036],
            [-0.81724176]]))
