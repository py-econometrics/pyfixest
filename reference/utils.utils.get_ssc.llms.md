# get_ssc

``` python
get_ssc(ssc_options, counts, *, vcov_type)
```

Compute the small sample adjustment factor and the degrees of freedom.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| ssc_options | Ssc | The options created via the ssc() function. | *required* |
| counts | DegreesOfFreedomCounts | Observation, coefficient, fixed-effect, and cluster counts. | *required* |
| vcov_type | str | The type of covariance matrix: “iid”, “hetero”, “HAC”, or “CRV”. | *required* |

## Returns

| Name | Type | Description |
|----|----|----|
|  | SmallSampleCorrection | The factor `adj` that multiplies the covariance matrix, the parameter count `df_k` it used, and the degrees of freedom `df_t` of the t distribution. |

## Examples

Called internally by the estimation functions. Use it directly to reproduce an adjustment factor by hand.

``` python
import pyfixest as pf
from pyfixest.utils.utils import DegreesOfFreedomCounts, get_ssc

# cluster-robust adjustment: 1000 observations, 3 coefficients, 20 clusters
counts = DegreesOfFreedomCounts(N=1000, k=3, k_fe=0, n_fe=0, G=20)
get_ssc(pf.ssc(), counts, vcov_type="CRV")
```

    SmallSampleCorrection(adj=1.0547431768991185, df_k=3, df_t=19)

Configure the behaviour with [ssc()](../reference/utils.utils.ssc.llms.md). See [On Small Sample Corrections](../explanation/ssc.llms.md) for the formulas.
