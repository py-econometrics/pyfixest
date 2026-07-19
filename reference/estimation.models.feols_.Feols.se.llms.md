# Feols.se

``` python
se()
```

Coefficient standard errors as a pandas Series.

Returns the `Std. Error` column of `tidy()`. The values depend on the variance estimator of the model, which can be changed via `vcov()`.

## Returns

| Name | Type          | Description                                   |
|------|---------------|-----------------------------------------------|
|      | pandas.Series | Standard errors, indexed by coefficient name. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.se()
```

    Coefficient
    X1    0.066373
    X2    0.017596
    Name: Std. Error, dtype: float64
