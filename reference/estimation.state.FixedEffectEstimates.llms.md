# FixedEffectEstimates

``` python
FixedEffectEstimates(fixed_effects, alpha, sumFE)
```

Fixed-effect estimates recovered by `fixef()`.

`fixef()` solves the least-squares problem in `alpha` under treatment coding, which drops a reference level of the second and every further fixed effect. With more than one fixed effect the individual levels are therefore identified only up to that normalization, while `sumFE` and contrasts within one fixed effect are invariant to it.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| fixed_effects | Mapping\[str, FixedEffect\] | Coefficient records keyed by Formulaic column name. `fixef()` returns their tidy frame with the original factor labels. | *required* |
| alpha | NDArray\[np.float64\] | Solution of the least-squares problem in the dummy-coded fixed effects, shape (n_fixed_effect_coefficients,), ordered as the columns of the fixed-effect dummy matrix. | *required* |
| sumFE | NDArray\[np.float64\] | Fixed-effect contribution of each observation, shape (n_rows,), in the units of the dependent variable. For GLMs it is on the scale of the linear predictor and excludes the offset. Named as in `fixest`. | *required* |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1", pf.get_data())
fit.fixef().head()
```

|     | variable | code | level | coefficient |
|-----|----------|------|-------|-------------|
| 0   | f1       | 0    | 0.0   | 0.488954    |
| 1   | f1       | 1    | 1.0   | 3.071600    |
| 2   | f1       | 2    | 2.0   | -1.092975   |
| 3   | f1       | 3    | 3.0   | 0.176623    |
| 4   | f1       | 4    | 4.0   | -1.218752   |

``` python
estimates = fit.fixef_estimates
estimates.sumFE[:5]
```

    array([ 3.07160009,  1.05093524,  2.11375001, -1.09297548, -1.09297548])
