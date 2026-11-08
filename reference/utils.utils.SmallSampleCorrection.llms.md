# SmallSampleCorrection

``` python
SmallSampleCorrection(adj, df_k, df_t)
```

Result of `get_ssc()`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| adj | float | Factor that multiplies the covariance matrix. | *required* |
| df_k | int | Parameters counted by the `k_adj` adjustment. | *required* |
| df_t | int or float | Degrees of freedom of the t reference distribution. | *required* |
