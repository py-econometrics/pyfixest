# Feols.fixef

``` python
fixef(atol=1e-06, btol=1e-06)
```

Compute the coefficients of (swept out) fixed effects for a regression model.

This method creates the following attributes: - `_alpha` (pd.DataFrame): A DataFrame with the estimated fixed effects. - `_sumFE` (np.array): An array with the sum of fixed effects for each observation (i = 1, …, N).

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | pd.DataFrame | A tidy DataFrame with columns `variable`, `code`, `level`, and `coefficient` containing the estimated fixed effects. |

## Examples

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fixed_effects = fit.fixef()
fixed_effects.head()
```

|     | variable | code | level | coefficient |
|-----|----------|------|-------|-------------|
| 0   | f1       | 15   | 15.0  | 1.887085    |
| 1   | f1       | 6    | 6.0   | -0.254456   |
| 2   | f1       | 1    | 1.0   | 3.066142    |
| 3   | f1       | 19   | 19.0  | 1.123039    |
| 4   | f1       | 13   | 13.0  | 2.013726    |
