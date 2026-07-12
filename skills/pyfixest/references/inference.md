# Inference

For regression models, choose vcov=iid, hetero, HC1, HC2, HC3, NW, DK, or a
cluster dictionary such as {CRV1: firm}. CRV1 supports two-way clustering as
{CRV1: firm + year}; CRV3 has model-specific support.

HAC requires explicit identifiers: NW and DK need vcov_kwargs with lag and
time_id; DK also needs panel_id. Quantile regression supports iid, nid,
heteroskedastic inference, HC aliases, and one-way CRV1, not HAC.

Use pf.ssc() for small-sample corrections. The default is k_adj=True,
k_fixef=nonnested, G_adj=True, G_df=min. State whether weights are aweights or
fweights and verify the estimator supports the combination.

Read pyfixest/docs/pages/tutorials/standard-errors.md and
pyfixest/docs/pages/explanation/ssc.md for the complete rules.
