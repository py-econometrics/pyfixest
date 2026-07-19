# Feols.resid

``` python
resid()
```

Fitted model residuals.

For weighted models the residuals are rescaled by the square root of the weights, so they are on the scale of the original dependent variable.

## Returns

| Name | Type | Description |
|----|----|----|
|  | np.ndarray | A np.ndarray with the residuals of the estimated regression model. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.resid()[:5]
```

    array([ 1.47475887,  0.30548207, -0.74805202, -0.77787528, -0.5459377 ])
