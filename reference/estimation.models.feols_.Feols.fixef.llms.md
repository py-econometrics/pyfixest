# Feols.fixef

``` python
fixef(atol=1e-06, btol=1e-06)
```

Compute the coefficients of (swept out) fixed effects for a regression model.

Publishes the estimates as `fixef_estimates`, a `FixedEffectEstimates` value holding the coefficient records, the dummy-coded solution `alpha`, and the per-observation contribution `sumFE`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A tidy DataFrame with columns `variable`, `code`, `level`, and `coefficient` containing the estimated fixed effects. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | NonConvergenceError | If LSQR stops at its iteration limit or because the problem is too ill-conditioned, before meeting `atol` and `btol`. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fixed_effects = fit.fixef()
fixed_effects.head()
```

|     | variable | code | level | coefficient |
|-----|----------|------|-------|-------------|
| 0   | f1       | 0    | 0.0   | 0.483757    |
| 1   | f1       | 1    | 1.0   | 3.066142    |
| 2   | f1       | 2    | 2.0   | -1.094787   |
| 3   | f1       | 3    | 3.0   | 0.331095    |
| 4   | f1       | 4    | 4.0   | -1.287253   |
