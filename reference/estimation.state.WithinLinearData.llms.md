# WithinLinearData

``` python
WithinLinearData(response, design)
```

Response and regressors after demeaning by the fixed effects.

Without fixed effects, the arrays retain their original values. When observation weights are supplied, demeaning uses those weights. These arrays have not been multiplied by square-root observation weights.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| response | NDArray\[np.float64\] | Demeaned response, shape (n_rows, 1). | *required* |
| design | NDArray\[np.float64\] | Demeaned regressors after removing collinear columns, shape (n_rows, n_coefficients). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.within_data.design.shape
```

    (997, 1)
