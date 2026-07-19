# ssc

``` python
ssc(k_adj=True, k_fixef='nonnested', G_adj=True, G_df='min', *args, **kwargs)
```

Set the small sample correction factor applied in `get_ssc()`.

## Parameters

| Name | Type | Description | Default |
|----|----|----|----|
| k_adj | bool | If True, applies a small sample correction of (N-1) / (N-k) where N is the number of observations and k is the number of estimated coefficients excluding any fixed effects projected out by either `feols()` or `fepois()`. | `True` |
| k_fixef | str | Equal to ‘none’: the fixed effects parameters are discarded when calculating k in (N-1) / (N-k). | `"none"` |
| G_adj | bool | If True, a cluster correction G/(G-1) is performed, with G the number of clusters. This argument is only relevant for clustered errors. | `True` |
| G_df | str | Controls how “G” is computed for multiway clustering if G_adj = True. Note that the covariance matrix in the multiway clustering case is of the form V = V_1 + V_2 - V_12. If “conventional”, then each summand G_i is multiplied with a small sample adjustment G_i / (G_i - 1). If “min”, all summands are multiplied with the same value, min(G) / (min(G) - 1). This argument is only relevant for clustered errors. | `"conventional"` |

## Details

The small sample correction choices mimic fixest’s behavior. For details, see https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html.

In general, if k_adj = True, we multiply the variance covariance matrix V with a small sample correction factor of (N-1) / (N-k), where N is the number of observations and k is the number of estimated coefficients.

If k_fixef = “none”, the fixed effects parameters are discarded when calculating k. This is the default behavior and currently the only option. Note that it is not r-fixest’s default behavior.

Hence if k_adj = True, the covariance matrix is computed as V = V x (N-1) / (N-k) for iid and heteroskedastic errors.

If k_adj = False, no small sample correction is applied of the type above is applied.

If G_adj = True, a cluster correction of G/(G-1) is performed, with G the number of clusters.

If k_adj = True and G_adj = True, V = V x (N - 1) / N - k) x G/(G-1) for cluster robust errors where G is the number of clusters.

If k_adj = False and G_adj = True, V = V x G/(G-1) for cluster robust errors, i.e. we drop the (N-1) / (N-k) factor. And if G_adj = False, no cluster correction is applied.

Things are slightly more complicated for multiway clustering. In this case, we compute the variance covariance matrix as V = V1 + V2 - V_12.

If G_adj = True and G_df = “conventional”, then V += \[V x G_i / (G_i - 1) for i in \[1, 2, 12\]\], i.e. each separate covariance matrix G_i is multiplied with a small sample adjustment G_i / (G_i - 1) corresponding to the number of clusters in the respective covariance matrix. This is the default behavior for clustered errors.

If G_df = “min”, then V += \[V x min(G) / (min(G) - 1) for i in \[1, 2, 12\]\].

## Returns

| Name | Type | Description |
|----|----|----|
|  | dict | A dictionary with encoded info on how to form small sample corrections |

## Examples

``` python
import pyfixest as pf

data = pf.get_data()

# turn off both the k and the G adjustment
fit = pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"})
fit_no_adj = pf.feols(
    "Y ~ X1 | f1", data, vcov={"CRV1": "f1"}, ssc=pf.ssc(k_adj=False, G_adj=False)
)

pf.etable([fit, fit_no_adj])
```

[TABLE]

Defaults follow `fixest`. See [On Small Sample Corrections](../explanation/ssc.llms.md) for details.
