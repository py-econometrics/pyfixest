# FixedEffectCounts

``` python
FixedEffectCounts(n_levels_by_fe=(), has_intercept_by_fe=(), n_slopes_by_fe=())
```

Level counts of the absorbed fixed effects in the estimation sample.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| n_levels_by_fe | tuple\[int, …\] | Number of levels of each fixed effect after sample filtering, in the order of `ModelDescription.fixed_effects`. A fixed effect left with a single level, e.g. by the GLM separation check, still counts. | `()` |
| has_intercept_by_fe | tuple\[bool, …\] | Whether each fixed effect has level-specific intercepts. Slope-only effects such as `f1[[z]]` have none. | `()` |
| n_slopes_by_fe | tuple\[int, …\] | Number of varying-slope covariates of each fixed effect; `f1[z1, z2]` has two. | `()` |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 | f1 + f2", pf.get_data())
counts = fit.fixef_counts
counts.n_levels_by_fe, counts.n_fixef, counts.n_levels, counts.fixef_dof
```

    ((30, 30), 2, 60, 59)

## Attributes

| Name | Description |
|----|----|
| [FixedEffectCounts.fixef_dof](#pyfixest.estimation.state.FixedEffectCounts.fixef_dof) | Degrees of freedom consumed by the fixed effects. |
| [FixedEffectCounts.n_fixef](#pyfixest.estimation.state.FixedEffectCounts.n_fixef) | Number of fixed effects with intercepts; `Y ~ X \| f1 + f2[[z]]` has one. |
| [FixedEffectCounts.n_levels](#pyfixest.estimation.state.FixedEffectCounts.n_levels) | Levels summed over the fixed effects with intercepts. |
| [FixedEffectCounts.n_slope_coefficients](#pyfixest.estimation.state.FixedEffectCounts.n_slope_coefficients) | Number of varying-slope coefficients, one per level and slope. |
