# Feols.get_performance

``` python
get_performance()
```

Get Goodness-of-Fit measures.

Compute multiple additional measures commonly reported with linear regression output, including R-squared and adjusted R-squared. Note that variables with the suffix \_within use demeaned dependent variables Y, while variables without do not or are invariant to demeaning.

## Returns

| Name | Type | Description |
|----|----|----|
|  | None | The measures are stored on the model object rather than returned. |

## Notes

Sets the attributes `_rmse`, `_r2`, `_adj_r2`, `_r2_within`, and `_adj_r2_within`. The `_within` variants are computed on the demeaned dependent variable and are only defined for models with fixed effects.

## Examples

The estimation functions call this during fitting, so the measures are available on any fitted model.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.get_performance()

fit._r2, fit._adj_r2, fit._r2_within
```

    (np.float64(0.4889939082821597),
     np.float64(0.47257816854821877),
     np.float64(0.23875790489544113))
