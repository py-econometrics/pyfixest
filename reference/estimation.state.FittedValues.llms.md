# FittedValues

``` python
FittedValues(link, response)
```

In-sample predictions / fitted values on link and the response scale.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| link | NDArray\[np.float64\] | Prediction on the scale of the linear predictor, including the fixed-effect contribution and any offset, shape (n_rows,). | *required* |
| response | NDArray\[np.float64\] | Prediction on the scale of the dependent variable, `E(Y\|X)`, shape (n_rows,). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.fitted_values.response[:3]
```

    array([ 2.12216361, -0.84794299,  0.21486527])
