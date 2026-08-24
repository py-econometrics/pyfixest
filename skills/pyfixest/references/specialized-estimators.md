# Specialized estimators and causal workflows

Choose the public entry point that matches the design:

- `pf.fepois(...)` or `pf.feglm(family="poisson", ...)` for Poisson models.
- `pf.feglm(...)` for supported GLM families.
- `pf.quantreg(...)` for quantile regression; do not assume fixed-effect or IV
  features from `feols` carry over.
- `pf.event_study(...)`, `pf.did2s(...)`, and `pf.lpdid(...)` for supported
  difference-in-differences workflows.
- `pf.panelview(...)` to inspect panel treatment timing.
- `pf.bonferroni(...)`, `pf.rwolf(...)`, and `pf.wyoung(...)` for supported
  multiple-testing adjustments.

Start with the bundled
[difference-in-differences tutorial](../../../pyfixest/docs/pages/tutorials/difference-in-differences.md),
[instrumental-variables tutorial](../../../pyfixest/docs/pages/tutorials/instrumental-variables.md),
or estimator-specific tutorial linked from the core API reference.

Specialized estimators differ in supported weights, covariance estimators,
fixed effects, prediction, and post-estimation methods. Check their installed
signatures and docs rather than transferring `feols` arguments by analogy.
