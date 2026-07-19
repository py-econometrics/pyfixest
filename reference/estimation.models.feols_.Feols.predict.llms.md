# Feols.predict

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

Predict values of the model on new data.

Return a flat np.array with predicted values of the regression model. If new fixed effect levels are introduced in `newdata`, predicted values for such observations will be set to NaN.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| newdata | DataFrameType | A narwhals compatible DataFrame (polars, pandas, duckdb, etc). If None (default), the data used for fitting the model is used. | `None` |
| type | str | The type of prediction to be computed. Can be either “response” (default) or “link”. For linear models, both are identical. | `'link'` |
| atol | Float | Stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| btol | Float | Another stopping tolerance for scipy.sparse.linalg.lsqr(). See https://docs.scipy.org/doc/ scipy/reference/generated/scipy.sparse.linalg.lsqr.html | `1e-6` |
| type | PredictionType | The type of prediction to be made. Can be either ‘link’ or ‘response’. Defaults to ‘link’. ‘link’ and ‘response’ lead to identical results for linear models. | `'link'` |
| se_fit | bool \| None | If True, the standard error of the prediction is computed. Only feasible for models without fixed effects. GLMs are not supported. Defaults to False. | `False` |
| interval | PredictionErrorOptions \| None | The type of interval to compute. Can be either ‘prediction’ or None. | `None` |
| alpha | float | The alpha level for the confidence interval. Defaults to 0.05. Only used if interval = “prediction” is not None. | `0.05` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | Union\[np.ndarray, pd.DataFrame\] | Returns a pd.Dataframe with columns “fit”, “se_fit” and CIs if argument “interval=prediction”. Otherwise, returns a np.ndarray with the predicted values of the model or the prediction standard errors if argument “se_fit=True”. |

## Examples

In-sample predictions:

``` python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data)
fit.predict()[:5]
```

    array([ 1.84475454, -0.17106206,  0.46970178, -0.74191438, -1.52651336])

Pass `newdata` to predict out of sample. Fixed effect levels that do not appear in the estimation sample return missing values.

``` python
fit.predict(newdata=data.head())
```

    array([ 1.80731416,         nan,         nan,  1.84475484, -0.17106194])
