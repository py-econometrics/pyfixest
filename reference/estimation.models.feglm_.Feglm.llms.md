# Feglm

``` python
Feglm(
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
    solver,
    family,
    demeaner=None,
    lookup_preconditioner=None,
    store_data=True,
    copy_data=True,
    lean=False,
    sample_split_var=None,
    sample_split_value=None,
    separation_check=None,
    context=0,
    accelerate=True,
)
```

Base class for the estimation of a fixed-effects GLM model.

Returned by [feglm()](../reference/estimation.api.feglm.feglm.llms.md). Fixed effects are handled via iteratively reweighted least squares with demeaning, following Stammann (2018), [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815). The family is set with the `family` argument and implemented by a subclass. `poisson` dispatches to [Fepois](../reference/estimation.models.fepois_.Fepois.llms.md).

## Examples

``` python
import numpy as np
import pyfixest as pf

data = pf.get_data()
data["Y_bin"] = np.where(data["Y"] > 0, 1, 0)

fit = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit")
fit.tidy()
```

|             | Estimate  | Std. Error | t value   | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|-----------|--------------|-----------|-----------|
| Coefficient |           |            |           |              |           |           |
| X1          | -1.016344 | 0.109325   | -9.296552 | 0.000000e+00 | -1.230617 | -0.802071 |
| X2          | -0.165899 | 0.028450   | -5.831291 | 5.500026e-09 | -0.221660 | -0.110139 |

Coefficients are on the link scale. For effects on the response scale, use `marginaleffects`. See the [marginal effects guide](../how-to/marginaleffects.llms.md).

``` python
from marginaleffects import avg_slopes

avg_slopes(fit, variables="X1")
```

shape: (1, 3)

| term | contrast | estimate  |
|------|----------|-----------|
| str  | str      | f64       |
| "X1" | "dY/dX"  | -1.016344 |

## Methods

| Name | Description |
|----|----|
| [Feglm.get_fit](#pyfixest.estimation.models.feglm_.Feglm.get_fit) | Fit the GLM via IRLS and write results onto self.\* attributes. |
| [Feglm.predict](#pyfixest.estimation.models.feglm_.Feglm.predict) | Return predicted values from regression model. |
| [Feglm.prepare_model_matrix](#pyfixest.estimation.models.feglm_.Feglm.prepare_model_matrix) | Prepare model inputs for estimation. |
| [Feglm.resid](#pyfixest.estimation.models.feglm_.Feglm.resid) | Return residuals from a fitted GLM. |
| [Feglm.residualize](#pyfixest.estimation.models.feglm_.Feglm.residualize) | Residualize v and X by flist using weights. |
| [Feglm.to_array](#pyfixest.estimation.models.feglm_.Feglm.to_array) | Turn estimation DataFrames to np arrays. |

### Feglm.get_fit

``` python
get_fit()
```

Fit the GLM via IRLS and write results onto self.\* attributes.

### Feglm.predict

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

### Feglm.prepare_model_matrix

``` python
prepare_model_matrix()
```

Prepare model inputs for estimation.

### Feglm.resid

``` python
resid(type='response')
```

Return residuals from a fitted GLM.

#### Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| type | str | The type of residuals to return. Either “response” (default) or “working”. | `'response'` |

#### Returns

| Name | Type       | Description                                |
|------|------------|--------------------------------------------|
|      | np.ndarray | A flat array with the requested residuals. |

### Feglm.residualize

``` python
residualize(v, X, flist, weights, tol)
```

Residualize v and X by flist using weights.

### Feglm.to_array

``` python
to_array()
```

Turn estimation DataFrames to np arrays.
