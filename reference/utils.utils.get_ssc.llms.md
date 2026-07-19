# get_ssc

``` python
get_ssc(
    ssc_dict,
    N,
    k,
    k_fe,
    k_fe_nested,
    n_fe,
    n_fe_fully_nested,
    G,
    vcov_sign,
    vcov_type,
)
```

Compute small sample adjustment factors.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| ssc_dict | dict | A dictionary created via the ssc() function. | *required* |
| N | int | The number of observations. | *required* |
| k | int | The number of estimated parameters (as in the first part of the model formula) | *required* |
| k_fe | int | The number of estimated fixed effects (as specified in the second part of the model formula). | *required* |
| k_fe_nested | int | The number of estimated fixed effects nested within clusters. | *required* |
| n_fe | int | The number of fixed effects in the model. I.e. ‘Y ~ X1 \| f1 + f2’ has 2 fixed effects. | *required* |
| n_fe_fully_nested | int | The number of fixed effects that are fully nested within clusters. | *required* |
| G | int | The number of clusters. | *required* |
| vcov_sign | array - like | A vector that helps create the covariance matrix. | *required* |
| vcov_type | str | The type of covariance matrix. Must be one of “iid”, “hetero”, “HAC”, or “CRV”. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | tuple of np.ndarray and int | A small sample adjustment factor and the effective number of coefficients k used in the adjustment. |

## Raises

| Name | Type | Description |
|----|----|----|
|  | ValueError | If vcov_type is not “iid”, “hetero”, or “CRV”, or if G_df is neither “conventional” nor “min”. |

## Examples

Called internally by the estimation functions. Use it directly to reproduce an adjustment factor by hand.

``` python
import pyfixest as pf
from pyfixest.utils.utils import get_ssc

# cluster-robust adjustment: 1000 observations, 3 coefficients, 20 clusters
adj, k_used, G_used = get_ssc(
    ssc_dict=pf.ssc(),
    N=1000,
    k=3,
    k_fe=0,
    k_fe_nested=0,
    n_fe=0,
    n_fe_fully_nested=0,
    G=20,
    vcov_sign=1,
    vcov_type="CRV",
)
adj, k_used, G_used
```

    (array([1.05474318]), 3, np.int64(19))

Configure the behaviour with [ssc()](../reference/utils.utils.ssc.llms.md). See [On Small Sample Corrections](../explanation/ssc.llms.md) for the formulas.
