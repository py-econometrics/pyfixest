# Fepois

``` python
Fepois(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    lookup_demeaned_data,
    tol,
    maxiter,
    solver='np.linalg.solve',
    demeaner=None,
    lookup_preconditioner=None,
    context=0,
    store_data=True,
    copy_data=True,
    lean=False,
    sample_split_var=None,
    sample_split_value=None,
    separation_check=None,
    offset=None,
)
```

Estimate a Poisson regression model.

Non user-facing class to estimate a Poisson regression model via Iterated Weighted Least Squares (IWLS).

Inherits from the Feglm class. Users should not directly instantiate this class, but rather use the [fepois()](../reference/estimation.api.fepois.fepois.llms.md) function. Note that no demeaning is performed in this class: demeaning is performed in the FixestMulti class (to allow for caching of demeaned variables for multiple estimation).

The method implements the algorithm from Stata’s `ppmlhdfe` module.

## Attributes

| Name | Type | Description |
|----|----|----|
| \_Y | np.ndarray | The demeaned dependent variable, a two-dimensional numpy array. |
| \_X | np.ndarray | The demeaned independent variables, a two-dimensional numpy array. |
| \_fe | np.ndarray | Fixed effects, a two-dimensional numpy array or None. |
| weights | np.ndarray | Weights, a one-dimensional numpy array or None. |
| coefnames | list\[str\] | Names of the coefficients in the design matrix X. |
| drop_singletons | bool | Whether to drop singleton fixed effects. |
| collin_tol | float | Tolerance level for the detection of collinearity. |
| maxiter | Optional\[int\], default=25 | Maximum number of iterations for the IRLS algorithm. |
| tol | Optional\[float\], default=1e-08 | Tolerance level for the convergence of the IRLS algorithm. |
| solver | str, optional. | The solver to use for the regression. Can be “np.linalg.lstsq”, “np.linalg.solve”, “scipy.linalg.solve” and “scipy.sparse.linalg.lsqr”. Defaults to “scipy.linalg.solve”. |
| demeaner | Optional\[AnyDemeaner\] | Resolved typed demeaner configuration. |
| fixef_tol | float, default = 1e-06. | Tolerance level for the convergence of the demeaning algorithm. |
| context | int or Mapping\[str, Any\] | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. |
| weights_name | Optional\[str\] | Name of the weights variable. |
| weights_type | Optional\[str\] | Type of weights variable. |
| \_data | pd.DataFrame | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`. |

## Examples

`Fepois` is returned by [fepois()](../reference/estimation.api.fepois.fepois.llms.md) and is not constructed directly. Post-estimation methods are inherited from [Feols](../reference/estimation.models.feols_.Feols.llms.md).

``` python
import pyfixest as pf

data = pf.get_data(model="Fepois")
fit = pf.fepois("Y ~ X1 + X2 | f1", data)

fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|) | 2.5%      | 97.5%    |
|-------------|-----------|------------|-----------|-------------|-----------|----------|
| Coefficient |           |            |           |             |           |          |
| X1          | 0.001864  | 0.040712   | 0.045784  | 0.963482    | -0.077930 | 0.081658 |
| X2          | -0.014261 | 0.010903   | -1.307958 | 0.190888    | -0.035631 | 0.007109 |

Coefficients are on the log scale. Exponentiating gives incidence rate ratios.

``` python
import numpy as np

np.exp(fit.coef())
```

    Coefficient
    X1    1.001866
    X2    0.985840
    Name: Estimate, dtype: float64

## Methods

| Name | Description |
|----|----|
| [Fepois.get_fit](#pyfixest.estimation.models.fepois_.Fepois.get_fit) | Fit via Feglm IRLS, then add Poisson-specific post-fit summary stats. |
| [Fepois.predict](#pyfixest.estimation.models.fepois_.Fepois.predict) | Return predicted values from regression model. |

### Fepois.get_fit

``` python
get_fit()
```

Fit via Feglm IRLS, then add Poisson-specific post-fit summary stats.

### Fepois.predict

``` python
predict(
    newdata=None,
    atol=1e-06,
    btol=1e-06,
    type='link',
    se_fit=False,
    interval=None,
    alpha=0.05,
)
```

Return predicted values from regression model.

Return a flat np.array with predicted values of the regression model. If new fixed effect levels are introduced in `newdata`, predicted values for such observations will be set to NaN.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| newdata | Union\[None, pd.DataFrame\] | A pd.DataFrame with the new data, to be used for prediction. If None (default), uses the data used for fitting the model. | `None` |
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| type | str | The type of prediction to be computed. Can be either “response” (default) or “link”. If type=“response”, the output is at the level of the response variable, i.e., it is the expected predictor E(Y\|X). If “link”, the output is at the level of the explanatory variables, i.e., the linear predictor X @ beta. | `'link'` |
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| se_fit | bool \| None | If True, the standard error of the prediction is computed. Only feasible for models without fixed effects. GLMs are not supported. Defaults to False. | `False` |
| interval | PredictionErrorOptions \| None | The type of interval to compute. Can be either ‘prediction’ or None. | `None` |
| alpha | float | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = “prediction” is not None. | `0.05` |

#### Returns

| Name | Type | Description |
|----|----|----|
|  | Union\[np.ndarray, pd.DataFrame\] | Returns a pd.Dataframe with columns “fit”, “se_fit” and CIs if argument “interval=prediction”. Otherwise, returns a np.ndarray with the predicted values of the model or the prediction standard errors if argument “se_fit=True”. |
