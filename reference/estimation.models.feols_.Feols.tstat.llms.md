# Feols.tstat

``` python
tstat()
```

Coefficient t-statistics as a pandas Series.

Returns the `t value` column of `tidy()`, i.e. each estimate divided by its standard error.

## Returns

| Name | Type          | Description                                |
|------|---------------|--------------------------------------------|
|      | pandas.Series | t-statistics, indexed by coefficient name. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.tstat()
```

    Coefficient
    X1   -14.305943
    X2    -9.901590
    Name: t value, dtype: float64
