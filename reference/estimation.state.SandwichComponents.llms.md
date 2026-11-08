# SandwichComponents

``` python
SandwichComponents(scores, hessian, bread)
```

Scores, Hessian, and bread of a fitted model’s sandwich covariance.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| scores | NDArray\[np.float64\] | Weighted scores `W X * u`, shape (n_rows, n_coefficients). | *required* |
| hessian | NDArray\[np.float64\] | Weighted cross-product `X' W X`, shape (n_coefficients, n_coefficients). | *required* |
| bread | NDArray\[np.float64\] | Inverse of `hessian`, shape (n_coefficients, n_coefficients). | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.sandwich.bread
```

    array([[0.00156999]])
