# FixedEffect

``` python
FixedEffect(formulaic_column, name, observed_codes, level_values, coefficients)
```

Columnar coefficient data belonging to one fixed effect.

## Attributes

| Name | Type | Description |
|----|----|----|
| formulaic_column | str | Internal column name in the Formulaic spec. Its factor metadata retains the original parsed term; `ModelMatrix.fixed_effects` exposes the original labels as column names. |
| name | str | User-facing fixed-effect name. Interacted variables are joined with `:`, for example `firm:year`. |
| observed_codes | NDArray\[np.int64\] | Integer codes identifying fixed-effect levels observed in the estimation sample after singleton removal. |
| level_values | tuple\[NDArray\[Any\], …\] | Original level values, stored as one array per fixed-effect component. Every array is aligned with `observed_codes`. |
| coefficients | NDArray\[np.float64\] | Estimated coefficient indexed by fixed-effect code. Omitted reference levels have coefficient zero. Levels removed before estimation have coefficient `NaN`. |

## Methods

| Name | Description |
|----|----|
| [FixedEffect.level_labels](#pyfixest.estimation.state.FixedEffect.level_labels) | Combine values of fixed effect levels into comma-separated string. |

### FixedEffect.level_labels

``` python
level_labels()
```

Combine values of fixed effect levels into comma-separated string.
