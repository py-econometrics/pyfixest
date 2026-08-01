# factor_interaction

``` python
factor_interaction(data, var2=None, *, ref=None, ref2=None, bin=None, bin2=None)
```

Fixest-style i() operator for categorical encoding with interactions.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| data | array - like | The categorical variable to encode. | *required* |
| var2 | array - like | Optional second variable to interact with (continuous or categorical). | `None` |
| ref | Hashable | Reference level to drop from `data`. | `None` |
| ref2 | Hashable | Reference level to drop from `var2` (only if `var2` is categorical). | `None` |
| bin | dict | Mapping of `new_level -> [old_levels]` for binning `data`. | `None` |
| bin2 | dict | Mapping of `new_level -> [old_levels]` for binning `var2`. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | FactorValues | The encoded factor values, ready for use in a formulaic model matrix. |

## Examples

Implements the `i()` operator and is used by writing `i(...)` in a formula rather than by calling it directly. Expands a categorical variable into indicators, optionally dropping a reference level and interacting it with a second variable. Commonly used for event study specifications. See the [formula syntax tutorial](../tutorials/formula-syntax.llms.md).

``` python
import pyfixest as pf

data = pf.get_data()

fit = pf.feols("Y ~ i(f1, ref=0)", data)
fit.tidy().head()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| Intercept   | -0.486859 | 0.319489   | -1.523869 | 1.278682e-01 | -1.113830 | 0.140112  |
| f1::1.0     | 2.544288  | 0.430799   | 5.905979  | 4.848718e-09 | 1.698881  | 3.389695  |
| f1::2.0     | -1.634681 | 0.451825   | -3.617948 | 3.123025e-04 | -2.521351 | -0.748011 |
| f1::3.0     | -0.463981 | 0.465731   | -0.996241 | 3.193818e-01 | -1.377939 | 0.449978  |
| f1::4.0     | -1.762712 | 0.455041   | -3.873741 | 1.143851e-04 | -2.655693 | -0.869731 |

Interacting with a continuous variable gives group-specific slopes.

``` python
pf.feols("Y ~ i(f1, X2, ref=0)", data).tidy().head()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|-------------|-----------|-----------|
| Coefficient |           |            |           |             |           |           |
| Intercept   | -0.163916 | 0.071887   | -2.280195 | 0.022813    | -0.304987 | -0.022844 |
| f1::1.0:X2  | -0.118795 | 0.096590   | -1.229891 | 0.219037    | -0.308346 | 0.070755  |
| f1::2.0:X2  | -0.193556 | 0.106548   | -1.816610 | 0.069586    | -0.402647 | 0.015535  |
| f1::3.0:X2  | -0.173543 | 0.114664   | -1.513488 | 0.130482    | -0.398562 | 0.051476  |
| f1::4.0:X2  | -0.055993 | 0.108040   | -0.518260 | 0.604395    | -0.268012 | 0.156027  |

## Notes

Naming convention (matches R fixest)::

    i(cyl)            -> cyl::4, cyl::6, cyl::8
    i(cyl, ref=4)     -> cyl::6, cyl::8
    i(cyl, wt)        -> cyl::4:wt, cyl::6:wt, cyl::8:wt
    i(cyl, wt, ref=4) -> cyl::6:wt, cyl::8:wt
