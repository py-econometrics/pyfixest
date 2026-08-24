# Standard errors and inference

Choose inference explicitly when it matters, and verify estimator-specific
support in the installed API. The bundled
[small-sample-correction explanation](../../../pyfixest/docs/pages/explanation/ssc.md)
describes `pf.ssc(...)`.

Common `vcov` inputs include:

```python
pf.feols("Y ~ X1", data=df, vcov="iid")
pf.feols("Y ~ X1", data=df, vcov="hetero")  # HC1
pf.feols("Y ~ X1", data=df, vcov={"CRV1": "firm"})
pf.feols("Y ~ X1", data=df, vcov={"CRV1": "firm + year"})
```

- `"iid"` assumes independent, homoskedastic errors.
- `"hetero"` and `"HC1"` request HC1. HC2 and HC3 are not supported with
  fixed effects or IV.
- `{"CRV1": "cluster"}` requests cluster-robust inference. Add cluster names
  with `+` for multiway clustering.
- `{"CRV3": "cluster"}` requests the cluster jackknife. CRV3 is not supported
  for IV models.
- `"NW"` and `"DK"` request Newey-West and Driscoll-Kraay HAC inference.
  Supply the required time and panel identifiers through `vcov_kwargs`; consult
  the installed signature for exact requirements.

Models can recompute supported inference through `fit.vcov(...)`. For bootstrap
and design-based inference, inspect `wildboottest()` and `ritest()` rather than
treating them as ordinary covariance estimators. Never silently replace an
unsupported combination with a different standard error.
