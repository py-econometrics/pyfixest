# WithinIvData

``` python
WithinIvData(response, design, instruments, endogenous)
```

Response, regressors, and instruments after demeaning by the fixed effects.

`design` is the full structural regressor matrix, including the endogenous regressors. `instruments` is the full instrument matrix, including exogenous regressors that instrument themselves. Every array is demeaned using the observation weights when supplied, but is not multiplied by their square roots. Without fixed effects, the arrays retain their original values.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| response | NDArray\[np.float64\] | Demeaned response, shape (n_rows, 1). | *required* |
| design | NDArray\[np.float64\] | Demeaned regressors, including endogenous regressors, after removing collinear columns, shape (n_rows, n_coefficients). | *required* |
| instruments | NDArray\[np.float64\] | Demeaned instruments, including exogenous regressors, after removing collinear columns, shape (n_rows, n_instruments). | *required* |
| endogenous | NDArray\[np.float64\] | Demeaned endogenous regressor, shape (n_rows, 1). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", pf.get_data())
fit.within_data.instruments.shape
```

    (997, 2)
