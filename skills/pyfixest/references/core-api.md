# Core APIs

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
