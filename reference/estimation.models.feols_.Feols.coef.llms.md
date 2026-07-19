# Feols.coef

``` python
coef()
```

Estimated coefficients as a pandas Series.

Returns the `Estimate` column of `tidy()`.

## Returns

| Name | Type          | Description                                   |
|------|---------------|-----------------------------------------------|
|      | pandas.Series | Point estimates, indexed by coefficient name. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.coef()
```

    Coefficient
    X1   -0.949526
    X2   -0.174225
    Name: Estimate, dtype: float64
