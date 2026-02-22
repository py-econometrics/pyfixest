# estimation.models.fepois_.Fepois { #pyfixest.estimation.models.fepois_.Fepois }

```python
estimation.models.fepois_.Fepois(
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
    solver='np.linalg.solve',
    demeaner_backend='numba',
    context=0,
    store_data=True,
    copy_data=True,
    lean=False,
    sample_split_var=None,
    sample_split_value=None,
    separation_check=None,
)
```

Estimate a Poisson regression model.

Non user-facing class to estimate a Poisson regression model via Iterated
Weighted Least Squares (IWLS).

Inherits from the Feols class. Users should not directly instantiate this class,
but rather use the [fepois()](/reference/estimation.api.fepois.fepois.qmd) function.
Note that no demeaning is performed in this class: demeaning is performed in the
FixestMulti class (to allow for caching of demeaned variables for multiple estimation).

The method implements the algorithm from Stata's `ppmlhdfe` module.

## Attributes {.doc-section .doc-section-attributes}

| Name             | Type                             | Description                                                                                                                                                                                                                                                          |
|------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _Y               | np.ndarray                       | The demeaned dependent variable, a two-dimensional numpy array.                                                                                                                                                                                                      |
| _X               | np.ndarray                       | The demeaned independent variables, a two-dimensional numpy array.                                                                                                                                                                                                   |
| _fe              | np.ndarray                       | Fixed effects, a two-dimensional numpy array or None.                                                                                                                                                                                                                |
| weights          | np.ndarray                       | Weights, a one-dimensional numpy array or None.                                                                                                                                                                                                                      |
| coefnames        | list\[str\]                      | Names of the coefficients in the design matrix X.                                                                                                                                                                                                                    |
| drop_singletons  | bool                             | Whether to drop singleton fixed effects.                                                                                                                                                                                                                             |
| collin_tol       | float                            | Tolerance level for the detection of collinearity.                                                                                                                                                                                                                   |
| maxiter          | Optional\[int\], default=25      | Maximum number of iterations for the IRLS algorithm.                                                                                                                                                                                                                 |
| tol              | Optional\[float\], default=1e-08 | Tolerance level for the convergence of the IRLS algorithm.                                                                                                                                                                                                           |
| solver           | str, optional.                   | The solver to use for the regression. Can be "np.linalg.lstsq", "np.linalg.solve", "scipy.linalg.solve", "scipy.sparse.linalg.lsqr" and "jax". Defaults to "scipy.linalg.solve".                                                                                     |
| demeaner_backend | DemeanerBackendOptions.          | The backend used for demeaning.                                                                                                                                                                                                                                      |
| fixef_tol        | float, default = 1e-06.          | Tolerance level for the convergence of the demeaning algorithm.                                                                                                                                                                                                      |
| context          | int or Mapping\[str, Any\]       | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. |
| weights_name     | Optional\[str\]                  | Name of the weights variable.                                                                                                                                                                                                                                        |
| weights_type     | Optional\[str\]                  | Type of weights variable.                                                                                                                                                                                                                                            |
| _data            | pd.DataFrame                     | The data frame used in the estimation. None if arguments `lean = True` or `store_data = False`.                                                                                                                                                                      |

## Methods

| Name | Description |
| --- | --- |
| [get_fit](#pyfixest.estimation.models.fepois_.Fepois.get_fit) | Fit a Poisson Regression Model via Iterated Weighted Least Squares (IWLS). |
| [predict](#pyfixest.estimation.models.fepois_.Fepois.predict) | Return predicted values from regression model. |
| [prepare_model_matrix](#pyfixest.estimation.models.fepois_.Fepois.prepare_model_matrix) | Prepare model inputs for estimation. |
| [resid](#pyfixest.estimation.models.fepois_.Fepois.resid) | Return residuals from regression model. |
| [to_array](#pyfixest.estimation.models.fepois_.Fepois.to_array) | Turn estimation DataFrames to np arrays. |

### get_fit { #pyfixest.estimation.models.fepois_.Fepois.get_fit }

```python
estimation.models.fepois_.Fepois.get_fit()
```

Fit a Poisson Regression Model via Iterated Weighted Least Squares (IWLS).

#### Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description   |
|--------|--------|---------------|
|        | None   |               |

#### Attributes {.doc-section .doc-section-attributes}

| Name     | Type       | Description                                                                     |
|----------|------------|---------------------------------------------------------------------------------|
| beta_hat | np.ndarray | Estimated coefficients.                                                         |
| Y_hat    | np.ndarray | Estimated dependent variable.                                                   |
| u_hat    | np.ndarray | Estimated residuals.                                                            |
| weights  | np.ndarray | Weights (from the last iteration of the IRLS algorithm).                        |
| X        | np.ndarray | Demeaned independent variables (from the last iteration of the IRLS algorithm). |
| Z        | np.ndarray | Demeaned independent variables (from the last iteration of the IRLS algorithm). |
| Y        | np.ndarray | Demeaned dependent variable (from the last iteration of the IRLS algorithm).    |

### predict { #pyfixest.estimation.models.fepois_.Fepois.predict }

```python
estimation.models.fepois_.Fepois.predict(
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

### prepare_model_matrix { #pyfixest.estimation.models.fepois_.Fepois.prepare_model_matrix }

```python
estimation.models.fepois_.Fepois.prepare_model_matrix()
```

Prepare model inputs for estimation.

### resid { #pyfixest.estimation.models.fepois_.Fepois.resid }

```python
estimation.models.fepois_.Fepois.resid(type='response')
```

Return residuals from regression model.

#### Parameters {.doc-section .doc-section-parameters}

| Name   | Type   | Description                                                                            | Default      |
|--------|--------|----------------------------------------------------------------------------------------|--------------|
| type   | str    | The type of residuals to be computed. Can be either "response" (default) or "working". | `'response'` |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type       | Description                                              |
|--------|------------|----------------------------------------------------------|
|        | np.ndarray | A flat array with the residuals of the regression model. |

### to_array { #pyfixest.estimation.models.fepois_.Fepois.to_array }

```python
estimation.models.fepois_.Fepois.to_array()
```

Turn estimation DataFrames to np arrays.