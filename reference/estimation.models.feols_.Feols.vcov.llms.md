# Feols.vcov

``` python
vcov(vcov, vcov_kwargs=None, data=None)
```

Compute covariance matrices for an estimated regression model.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| vcov | Union\[str, dict\[str, str\]\] | A string or dictionary specifying the type of variance-covariance matrix to use for inference. If a string, it can be one of “iid”, “hetero”, “HC1”, “HC2”, “HC3”, “NW”, “DK”. If a dictionary, it should have the format {“CRV1”: “clustervar”} for CRV1 inference or {“CRV3”: “clustervar”} for CRV3 inference. Note that CRV3 inference is currently not supported for IV estimation. | *required* |
| vcov_kwargs | Optional\[dict\[str, any\]\] | Additional keyword arguments for the variance-covariance matrix. | `None` |
| data | DataFrameType \| None | The data used for estimation. If None, tries to fetch the data from the model object. Defaults to None. | `None` |

## Returns

| Name | Type | Description |
|----|----|----|
|  | Feols | An instance of class [Feols](../reference/estimation.models.feols_.Feols.llms.md) with updated inference. |

## Examples

Updates the variance estimator of a fitted model without refitting it. The model is modified in place and returned.

``` python
import pyfixest as pf

fit = pf.feols("Y ~ X1 + X2 | f1", pf.get_data())
fit.vcov("iid").tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|) | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|-------------|-----------|-----------|
| Coefficient |           |            |            |             |           |           |
| X1          | -0.949526 | 0.066373   | -14.305943 | 0.0         | -1.079777 | -0.819274 |
| X2          | -0.174225 | 0.017596   | -9.901590  | 0.0         | -0.208755 | -0.139695 |

``` python
# switch to cluster-robust inference
fit.vcov({"CRV1": "f1"}).tidy()
```

|             | Estimate  | Std. Error | t value    | Pr(\>\|t\|)  | 2.5%      | 97.5%     |
|-------------|-----------|------------|------------|--------------|-----------|-----------|
| Coefficient |           |            |            |              |           |           |
| X1          | -0.949526 | 0.066557   | -14.266314 | 1.221245e-14 | -1.085650 | -0.813401 |
| X2          | -0.174225 | 0.018409   | -9.464130  | 2.267890e-10 | -0.211876 | -0.136575 |

See [On Small Sample Corrections](../explanation/ssc.llms.md) for how the `ssc` adjustments interact with each estimator.
