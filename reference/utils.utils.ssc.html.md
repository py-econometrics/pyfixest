# utils.utils.ssc { #pyfixest.utils.utils.ssc }

```python
utils.utils.ssc(
    k_adj=True,
    k_fixef='nonnested',
    G_adj=True,
    G_df='min',
    *args,
    **kwargs,
)
```

Set the small sample correction factor applied in `get_ssc()`.

## Details {.doc-section .doc-section-details}

The small sample correction choices mimic fixest's behavior. For details, see
https://cran.r-project.org/web/packages/fixest/vignettes/standard_errors.html.

In general, if k_adj = True, we multiply the variance covariance matrix V with a
small sample correction factor of (N-1) / (N-k), where N is the number of
observations and k is the number of estimated coefficients.

If k_fixef = "none", the fixed effects parameters are discarded when
calculating k. This is the default behavior and currently the only
option. Note that it is not r-fixest's default behavior.

Hence if k_adj = True, the covariance matrix is computed as
V = V x (N-1) / (N-k) for iid and heteroskedastic errors.

If k_adj = False, no small sample correction is applied of the type
above is applied.

If G_adj = True, a cluster correction of G/(G-1) is performed,
with G the number of clusters.

If k_adj = True and G_adj = True, V = V x (N - 1) / N - k) x G/(G-1)
for cluster robust errors where G is the number of clusters.

If k_adj = False and G_adj = True, V = V x G/(G-1) for cluster robust
errors, i.e. we drop the (N-1) / (N-k) factor. And if G_adj = False,
no cluster correction is applied.

Things are slightly more complicated for multiway clustering. In this
case, we compute the variance covariance matrix as V = V1 + V2 - V_12.

If G_adj = True and G_df = "conventional", then
V += [V x G_i / (G_i - 1) for i in [1, 2, 12]], i.e. each separate
covariance matrix G_i is multiplied with a small sample adjustment
G_i / (G_i - 1) corresponding to the number of clusters in the
respective covariance matrix. This is the default behavior
for clustered errors.

If G_df = "min", then
V += [V x min(G) / (min(G) - 1) for i in [1, 2, 12]].

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                            |
|--------|--------|------------------------------------------------------------------------|
|        | dict   | A dictionary with encoded info on how to form small sample corrections |