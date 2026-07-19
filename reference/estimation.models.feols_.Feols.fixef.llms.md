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
|  | dict\[str, dict\[str, float\]\] | A dictionary with the estimated fixed effects. |

## Examples

Fixed effects are swept out during estimation and are not part of the coefficient table. `fixef()` computes them. The result is keyed by fixed effect term, then by level.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fe = fit.fixef()

fe.keys()
```

    dict_keys(['C(f1)'])

``` python
list(fe["C(f1)"].items())[:5]
```

    [('0.0', np.float64(0.4837574151832394)),
     ('1.0', np.float64(3.0661419612921605)),
     ('2.0', np.float64(-1.0947871507593956)),
     ('3.0', np.float64(0.33109523121756435)),
     ('4.0', np.float64(-1.2872533793542074))]
