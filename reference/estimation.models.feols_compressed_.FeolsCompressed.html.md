# estimation.models.feols_compressed_.FeolsCompressed { #pyfixest.estimation.models.feols_compressed_.FeolsCompressed }

```python
estimation.models.feols_compressed_.FeolsCompressed(
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
    solver='np.linalg.solve',
    demeaner_backend='numba',
    store_data=True,
    copy_data=True,
    lean=False,
    context=0,
    sample_split_var=None,
    sample_split_value=None,
    reps=None,
    seed=None,
)
```

Non-user-facing class for compressed regression with fixed effects.

See the paper "You only compress once" by Wong et al (https://arxiv.org/abs/2102.11297) for
details on regression compression.

## Parameters {.doc-section .doc-section-parameters}

| Name                 | Type                            | Description                                                                                                                                                                                                                                                          | Default             |
|----------------------|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| FixestFormula        | FixestFormula                   | The formula object.                                                                                                                                                                                                                                                  | _required_          |
| data                 | pd.DataFrame                    | The data.                                                                                                                                                                                                                                                            | _required_          |
| ssc_dict             | dict\[str, Union\[str, bool\]\] | The ssc dictionary.                                                                                                                                                                                                                                                  | _required_          |
| drop_singletons      | bool                            | Whether to drop columns with singleton fixed effects.                                                                                                                                                                                                                | _required_          |
| drop_intercept       | bool                            | Whether to include an intercept.                                                                                                                                                                                                                                     | _required_          |
| weights              | Optional\[str\]                 | The column name of the weights. None if no weights are used. For this method, weights needs to be None.                                                                                                                                                              | _required_          |
| weights_type         | Optional\[str\]                 | The type of weights. For this method, weights_type needs to be 'fweights'.                                                                                                                                                                                           | _required_          |
| collin_tol           | float                           | The tolerance level for collinearity.                                                                                                                                                                                                                                | _required_          |
| fixef_tol            | float                           | The tolerance level for the fixed effects.                                                                                                                                                                                                                           | _required_          |
| fixef_maxiter        | int                             | The maximum iterations for the demeaning algorithm.                                                                                                                                                                                                                  | _required_          |
| lookup_demeaned_data | dict\[str, pd.DataFrame\]       | The lookup table for demeaned data.                                                                                                                                                                                                                                  | _required_          |
| solver               | SolverOptions                   | The solver to use.                                                                                                                                                                                                                                                   | `'np.linalg.solve'` |
| store_data           | bool                            | Whether to store the data.                                                                                                                                                                                                                                           | `True`              |
| copy_data            | bool                            | Whether to copy the data.                                                                                                                                                                                                                                            | `True`              |
| lean                 | bool                            | Whether to keep memory-heavy objects as attributes or not.                                                                                                                                                                                                           | `False`             |
| context              | int or Mapping\[str, Any\]      | A dictionary containing additional context variables to be used by formulaic during the creation of the model matrix. This can include custom factorization functions, transformations, or any other variables that need to be available in the formula environment. | `0`                 |
| reps                 | int                             | The number of bootstrap repetitions. Default is 100. Only used for CRV1 inference, where a wild cluster bootstrap is used.                                                                                                                                           | `None`              |
| seed                 | Optional\[int\]                 | The seed for the random number generator. Only relevant for CRV1 inference, where a wild cluster bootstrap is used.                                                                                                                                                  | `None`              |

## Methods

| Name | Description |
| --- | --- |
| [demean](#pyfixest.estimation.models.feols_compressed_.FeolsCompressed.demean) | Compression 'handles demeaning' via Mundlak transform. |
| [predict](#pyfixest.estimation.models.feols_compressed_.FeolsCompressed.predict) | Compute predicted values. |
| [prepare_model_matrix](#pyfixest.estimation.models.feols_compressed_.FeolsCompressed.prepare_model_matrix) | Prepare model inputs for estimation. |
| [vcov](#pyfixest.estimation.models.feols_compressed_.FeolsCompressed.vcov) | Compute the variance-covariance matrix for the compressed regression. |

### demean { #pyfixest.estimation.models.feols_compressed_.FeolsCompressed.demean }

```python
estimation.models.feols_compressed_.FeolsCompressed.demean()
```

Compression 'handles demeaning' via Mundlak transform.

### predict { #pyfixest.estimation.models.feols_compressed_.FeolsCompressed.predict }

```python
estimation.models.feols_compressed_.FeolsCompressed.predict(
    newdata=None,
    atol=1e-06,
    btol=1e-06,
    type='link',
    se_fit=False,
    interval=None,
    alpha=0.05,
)
```

Compute predicted values.

#### Parameters {.doc-section .doc-section-parameters}

| Name     | Type                               | Description                                                                                                                                           | Default   |
|----------|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| newdata  | Optional\[DataFrameType\]          | The new data. If None, makes a prediction based on the uncompressed data set.                                                                         | `None`    |
| atol     | float                              | The absolute tolerance.                                                                                                                               | `1e-06`   |
| btol     | float                              | The relative tolerance.                                                                                                                               | `1e-06`   |
| type     | str                                | The type of prediction.                                                                                                                               | `'link'`  |
| se_fit   | Optional\[bool\]                   | If True, the standard error of the prediction is computed. Only feasible for models without fixed effects. GLMs are not supported. Defaults to False. | `False`   |
| interval | Optional\[PredictionErrorOptions\] | The type of interval to compute. Can be either 'prediction' or None.                                                                                  | `None`    |
| alpha    | float                              | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = "prediction" is not None.                                      | `0.05`    |

#### Returns {.doc-section .doc-section-returns}

| Name   | Type                              | Description                                                                                                                                                                                                                        |
|--------|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        | Union\[np.ndarray, pd.DataFrame\] | Returns a pd.Dataframe with columns "fit", "se_fit" and CIs if argument "interval=prediction". Otherwise, returns a np.ndarray with the predicted values of the model or the prediction standard errors if argument "se_fit=True". |

### prepare_model_matrix { #pyfixest.estimation.models.feols_compressed_.FeolsCompressed.prepare_model_matrix }

```python
estimation.models.feols_compressed_.FeolsCompressed.prepare_model_matrix()
```

Prepare model inputs for estimation.

### vcov { #pyfixest.estimation.models.feols_compressed_.FeolsCompressed.vcov }

```python
estimation.models.feols_compressed_.FeolsCompressed.vcov(
    vcov,
    vcov_kwargs=None,
    data=None,
)
```

Compute the variance-covariance matrix for the compressed regression.