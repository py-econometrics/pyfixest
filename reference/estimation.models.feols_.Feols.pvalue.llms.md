# Feols.pvalue

``` python
pvalue()
```

Coefficient p-values as a pandas Series.

Returns the `Pr(>|t|)` column of `tidy()`, for the two-sided null that a coefficient is zero.

## Returns

| Name | Type          | Description                            |
|------|---------------|----------------------------------------|
|      | pandas.Series | p-values, indexed by coefficient name. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.pvalue()
```

    Coefficient
    X1    0.0
    X2    0.0
    Name: Pr(>|t|), dtype: float64
