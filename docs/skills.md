---
title: "PyFixest agent skill"
description: "Progressive-disclosure skill for agents using PyFixest."
---

# PyFixest agent skill

<!-- Generated from skills/pyfixest; do not edit. -->

This page is generated from the skill bundled with the package. Installers
and downstream projects should copy the canonical skill directory, not this
rendered page.

Use the installed, version-matched documentation before relying on memory. Start
with pyfixest/docs/llms.txt or pyfixest/docs/index.md; use public functions
such as pyfixest.feols instead of internal modules.

## Workflow

1. Identify the estimator and whether the request needs fixed effects, IV,
   clustering/HAC, multiple estimation, or a specialized estimator.
2. Read the matching reference below, then confirm details in the installed docs
   page it names. Do not invent unsupported combinations.
3. Write a minimal public-API example. Use a seeded np.random.default_rng for
   synthetic data and retain the fitted result for post-estimation.
4. For an error, quote its received value and follow its local documentation
   pointer before changing the model specification.

## References

- Core APIs — estimator choice, result objects, and
  multiple estimation.
- Formula syntax — fixed effects, IV, i(), and
  multiple-estimation operators.
- Inference — vcov, clustering, HAC, weights, and SSC.
- Reporting — tidy results, tables, plots, and IV
  diagnostics.
- Specialized estimators — GLM, quantile,
  DiD/event-study, and multiple testing.
- Demeaners — choose MAP or LSMR backend and reuse
  preconditioners.
- Troubleshooting — formula, stored-data,
  optional-dependency, and unsupported-combination failures.

## Core APIs

Use import pyfixest as pf. Fit through these public entry points:

- pf.feols(fml, data, ...) for OLS, WLS, fixed effects, and IV.
- pf.fepois(fml, data, ...) for Poisson fixed-effects models.
- pf.feglm(fml, data, family=..., ...) for logit, probit, gaussian, or Poisson.
- pf.quantreg(fml, data, quantile=..., ...) for quantile regression; it does
  not support fixed-effects or IV formula parts.

One fit returns a result (Feols, Feiv, Fepois, a GLM subclass, or Quantreg).
Formula expansion, a quantile list, split, or fsplit returns FixestMulti. Use
fit.to_list() or fit.fetch_model(i) to select models; fit.tidy() combines their
coefficient tables.

Prefer store_data=True when later operations need original columns; use lean=True
only when the memory saving outweighs post-estimation features. Read
pyfixest/docs/pages/reference/estimation.api.feols.feols.md for the shared
estimation interface.

## Formula syntax

Use y ~ x1 + x2 for OLS, y ~ x1 | firm + year for fixed effects, and
y ~ x1 | firm + year | endogenous ~ instrument for IV. The IV part is
endogenous ~ excluded_instruments; use the three-part form when fixed effects
are present.

Use i(category, ref=...) for indicator expansion or i(category, variable) for
interactions. Use firm ^ year to create an interacted fixed effect.

For multiple estimation, use sw, sw0, csw, csw0, or mvsw; combine them
deliberately because they fan out to several models. Use split for one fit per
group and fsplit to include the full sample too.

Read pyfixest/docs/pages/tutorials/formula-syntax.md before translating a
complex R fixest formula.

## Inference

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

## Reporting and result APIs

Use fit.tidy(), fit.coef(), fit.se(), fit.tstat(), fit.pvalue(), and
fit.confint() for machine-readable results. Use fit.summary(), pf.etable,
pf.dtable, pf.coefplot, pf.iplot, and pf.qplot for presentation.

The reporting wrappers are explicit methods on single results and FixestMulti:
summary(), etable(), coefplot(), and iplot(). Pass type=df, md, tex, or another
documented output type to etable as needed.

For IV, call first_stage() before reading first_stage_model or
first_stage_f_statistic; call IV_Diag() before effective_f_statistic. Do not
rely on private underscore attributes.

Read pyfixest/docs/pages/tutorials/regression-tables.md and the installed Feols
or Feiv reference page for supported post-estimation methods.

## Specialized estimators

Use fepois or feglm for fixed-effects count and GLM models. feglm accepts the
documented families; offsets are supported only for Poisson. Validate
separation, weights, and fixed-effects support before generalizing an OLS
workflow.

Use quantreg with one quantile or a list of floats strictly between zero and
one. A list returns FixestMulti; pf.qplot visualizes the process. Do not use
fixed-effects or IV formula parts with quantile regression.

For difference-in-differences, use pf.event_study, pf.did2s, or pf.lpdid. Use
pf.panelview to inspect treatment timing. The packaged pyfixest.did/data/df_het.csv
fixture supports offline examples through importlib.resources.files(pyfixest.did).

For family-wise adjustments, use pf.bonferroni, pf.rwolf, or pf.wyoung. Read the
installed DiD, Poisson/GLM, and quantile tutorials before combining specialized
estimators with nonstandard inference.

## Demeaners

Use the default MapDemeaner() for typical high-dimensional fixed effects. Choose
MapDemeaner(backend=numba) only when its optional dependency is available. Use
LsmrDemeaner() for sparse or difficult fixed-effects designs; the optional torch
backend needs the corresponding extra and device support.

Pass a typed demeaner= configuration rather than deprecated loose backend or
tolerance arguments. The result exposes a reusable preconditioner for a later
model with the same fixed-effect design; do not reuse it across a different
design.

Read pyfixest/docs/pages/how-to/demeaner-backends.md for backend limits,
tolerances, preconditioners, and optional-dependency setup.

## Troubleshooting

Start with the received value in a PyFixest error. Formula errors usually need a
valid y ~ rhs | fixed_effects | endogenous ~ instruments form. Vcov errors
usually need a supported name, existing cluster columns, or HAC identifiers.

MissingStoredDataError means a post-estimation operation needs data removed by
store_data=False or lean=True. Refit with store_data=True; for eligible vcov()
calls, pass the original data explicitly.

Unsupported combinations are intentional: check fixed effects, IV, weights,
multiple estimation, and the model family before retrying. Install optional
dependencies only when selecting their paths, for example numba, lets-plot, or
torch.

Read pyfixest/docs/pages/troubleshooting.md first, then the formula, inference,
or demeaner reference named by the error.
