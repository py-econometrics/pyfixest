# Feglm

``` python
Feglm(
    FixestFormula,
    data,
    *,
    options,
    lookup_demeaned_data,
    lookup_preconditioner=None,
    sample_split=None,
)
```

Base class for the estimation of a fixed-effects GLM model.

Returned by [feglm()](../reference/estimation.api.feglm.feglm.llms.md). Fixed effects are handled via iteratively reweighted least squares with demeaning, following Stammann (2018), [arXiv:1707.01815](https://arxiv.org/pdf/1707.01815). The `family` argument selects a subclass that declares its `GlmFamily`. `poisson` dispatches to [Fepois](../reference/estimation.models.fepois_.Fepois.llms.md).

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

## Methods

| Name | Description |
|----|----|
| [Feglm.get_fit](#pyfixest.estimation.models.feglm_.Feglm.get_fit) | Fit the GLM via IRLS and write results onto self.\* attributes. |
| [Feglm.predict](#pyfixest.estimation.models.feglm_.Feglm.predict) | Return predicted values from regression model. |
| [Feglm.prepare_model_matrix](#pyfixest.estimation.models.feglm_.Feglm.prepare_model_matrix) | Prepare model inputs for estimation. |
| [Feglm.resid](#pyfixest.estimation.models.feglm_.Feglm.resid) | Return residuals from a fitted GLM. |
| [Feglm.residualize](#pyfixest.estimation.models.feglm_.Feglm.residualize) | Residualize v and X with current working weights and a reused preconditioner. |

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
| se_fit | bool \| None | Prediction standard errors are not supported for GLMs. Passing True raises NotImplementedError. Defaults to False. | `False` |
| interval | PredictionErrorOptions \| None | Prediction intervals are not supported for GLMs. Passing ‘prediction’ raises NotImplementedError. Defaults to None. | `None` |
| alpha | float | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = “prediction” is not None. | `0.05` |

#### Returns

| Name | Type       | Description                              |
|------|------------|------------------------------------------|
|      | np.ndarray | Predicted values on the requested scale. |

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

Residualize v and X with current working weights and a reused preconditioner.
