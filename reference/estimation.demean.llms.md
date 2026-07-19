# estimation.demean

``` python
estimation.demean(x, flist, weights, tol=1e-06, maxiter=10000)
```

Demean an array.

Workhorse for demeaning an input array `x` based on the specified fixed effects and weights via the alternating projections algorithm.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| x | numpy.ndarray | Input array of shape (n_samples, n_features). Needs to be of type float. | *required* |
| flist | numpy.ndarray | Array of shape (n_samples, n_factors) specifying the fixed effects. Needs to already be converted to integers. | *required* |
| weights | numpy.ndarray | Array of shape (n_samples,) specifying the weights. | *required* |
| tol | float | Tolerance criterion for convergence. Defaults to 1e-06. | `1e-06` |
| maxiter | int | Maximum number of iterations. Defaults to 10_000. | `10000` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | tuple\[numpy.ndarray, bool\] | A tuple containing the demeaned array of shape (n_samples, n_features) and a boolean indicating whether the algorithm converged successfully. |

## Examples

``` python
import numpy as np
import pyfixest as pf
from pyfixest.utils.dgps import get_blw
from pyfixest.estimation import demean
from formulaic import model_matrix

fml = "y ~ treat | state + year"

data = get_blw()
data.head()

Y, rhs = model_matrix(fml, data)
X = rhs[0].drop(columns="Intercept")
fe = rhs[1].drop(columns="Intercept")
YX = np.concatenate([Y, X], axis=1)

# to numpy
Y = Y.to_numpy()
X = X.to_numpy()
YX = np.concatenate([Y, X], axis=1)
fe = fe.to_numpy().astype(int)  # demean requires fixed effects as ints!

YX_demeaned, success = demean(YX, fe, weights = np.ones(YX.shape[0]))
Y_demeaned = YX_demeaned[:, 0]
X_demeaned = YX_demeaned[:, 1:]

print(np.linalg.lstsq(X_demeaned, Y_demeaned, rcond=None)[0])
print(pf.feols(fml, data).coef())
```

    [-6.75407097]
    Coefficient
    treat   -6.754071
    Name: Estimate, dtype: float64
