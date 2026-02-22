# estimation.internals.demean_.demean { #pyfixest.estimation.internals.demean_.demean }

```python
estimation.internals.demean_.demean(
    x,
    flist,
    weights,
    tol=1e-08,
    maxiter=100000,
)
```

Demean an array.

Workhorse for demeaning an input array `x` based on the specified fixed
effects and weights via the alternating projections algorithm.

## Parameters {.doc-section .doc-section-parameters}

| Name    | Type          | Description                                                                                                    | Default    |
|---------|---------------|----------------------------------------------------------------------------------------------------------------|------------|
| x       | numpy.ndarray | Input array of shape (n_samples, n_features). Needs to be of type float.                                       | _required_ |
| flist   | numpy.ndarray | Array of shape (n_samples, n_factors) specifying the fixed effects. Needs to already be converted to integers. | _required_ |
| weights | numpy.ndarray | Array of shape (n_samples,) specifying the weights.                                                            | _required_ |
| tol     | float         | Tolerance criterion for convergence. Defaults to 1e-08.                                                        | `1e-08`    |
| maxiter | int           | Maximum number of iterations. Defaults to 100_000.                                                             | `100000`   |

## Returns {.doc-section .doc-section-returns}

| Name   | Type                         | Description                                                                                                                                   |
|--------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
|        | tuple\[numpy.ndarray, bool\] | A tuple containing the demeaned array of shape (n_samples, n_features) and a boolean indicating whether the algorithm converged successfully. |

## Examples {.doc-section .doc-section-examples}

```{python}
import numpy as np
import pyfixest as pf
from pyfixest.utils.dgps import get_blw
from pyfixest.estimation.internals.demean_ import demean
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