# estimation.models.feglm_.Feglm { #pyfixest.estimation.models.feglm_.Feglm }

```python
estimation.models.feglm_.Feglm(
    FixestFormula,
    data,
    ssc_dict,
    drop_singletons,
    drop_intercept,
    weights,
    weights_type,
    collin_tol,
    fixef_tol,
    fixef_maxiter,
    lookup_demeaned_data,
    tol,
    maxiter,
    solver,
    demeaner_backend='numba',
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

Abstract base class for the estimation of a fixed-effects GLM model.

## Methods

| Name | Description |
| --- | --- |
| [get_fit](#pyfixest.estimation.models.feglm_.Feglm.get_fit) | Fit the GLM model via iterated weighted least squares. |
| [predict](#pyfixest.estimation.models.feglm_.Feglm.predict) | Return predicted values from regression model. |
| [prepare_model_matrix](#pyfixest.estimation.models.feglm_.Feglm.prepare_model_matrix) | Prepare model inputs for estimation. |
| [residualize](#pyfixest.estimation.models.feglm_.Feglm.residualize) | Residualize v and X by flist using weights. |
| [to_array](#pyfixest.estimation.models.feglm_.Feglm.to_array) | Turn estimation DataFrames to np arrays. |

### get_fit { #pyfixest.estimation.models.feglm_.Feglm.get_fit }

```python
estimation.models.feglm_.Feglm.get_fit()
```

Fit the GLM model via iterated weighted least squares.

The implementation follows ideas developed in
- Bergé (2018): https://ideas.repec.org/p/luc/wpaper/18-13.html
- Correia, Guimaraes, Zylkin (2019): https://journals.sagepub.com/doi/pdf/10.1177/1536867X20909691
- Stamann (2018): https://arxiv.org/pdf/1707.01815

### predict { #pyfixest.estimation.models.feglm_.Feglm.predict }

```python
estimation.models.feglm_.Feglm.predict(
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

Return a flat np.array with predicted values of the regression model.
If new fixed effect levels are introduced in `newdata`, predicted values
for such observations
will be set to NaN.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type                               | Description                                                                                                                                                                                                                                                                                                        | Default   |
|----------|------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| newdata  | Union\[None, pd.DataFrame\]        | A pd.DataFrame with the new data, to be used for prediction. If None (default), uses the data used for fitting the model.                                                                                                                                                                                          | `None`    |
| atol     | Float                              | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/     scipy/reference/generated/scipy.sparse.linalg.lsqr.html                                                                                                                                                                     | `1e-6`    |
| btol     | Float                              | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/     scipy/reference/generated/scipy.sparse.linalg.lsqr.html                                                                                                                                                             | `1e-6`    |
| type     | str                                | The type of prediction to be computed. Can be either "response" (default) or "link". If type="response", the output is at the level of the response variable, i.e., it is the expected predictor E(Y\|X). If "link", the output is at the level of the explanatory variables, i.e., the linear predictor X @ beta. | `'link'`  |
| atol     | Float                              | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html                                                                                                                                                                          | `1e-6`    |
| btol     | Float                              | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html                                                                                                                                                                  | `1e-6`    |
| se_fit   | Optional\[bool\]                   | If True, the standard error of the prediction is computed. Only feasible for models without fixed effects. GLMs are not supported. Defaults to False.                                                                                                                                                              | `False`   |
| interval | Optional\[PredictionErrorOptions\] | The type of interval to compute. Can be either 'prediction' or None.                                                                                                                                                                                                                                               | `None`    |
| alpha    | float                              | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = "prediction" is not None.                                                                                                                                                                                                   | `0.05`    |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                              | Description                                                                                                                                                                                                                        |
|--------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | Union\[np.ndarray, pd.DataFrame\] | Returns a pd.Dataframe with columns "fit", "se_fit" and CIs if argument "interval=prediction". Otherwise, returns a np.ndarray with the predicted values of the model or the prediction standard errors if argument "se_fit=True". |

### prepare_model_matrix { #pyfixest.estimation.models.feglm_.Feglm.prepare_model_matrix }

```python
estimation.models.feglm_.Feglm.prepare_model_matrix()
```

Prepare model inputs for estimation.

### residualize { #pyfixest.estimation.models.feglm_.Feglm.residualize }

```python
estimation.models.feglm_.Feglm.residualize(v, X, flist, weights, tol, maxiter)
```

Residualize v and X by flist using weights.

### to_array { #pyfixest.estimation.models.feglm_.Feglm.to_array }

```python
estimation.models.feglm_.Feglm.to_array()
```

Turn estimation DataFrames to np arrays.