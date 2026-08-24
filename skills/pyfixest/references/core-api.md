# Core API and result objects

Import the public package as `import pyfixest as pf`. The main estimation entry
points are:

- `pf.feols(...)` for OLS, WLS, high-dimensional fixed effects, and IV.
- `pf.fepois(...)` for Poisson regression, including fixed effects and offsets.
- `pf.feglm(...)` for Gaussian, logit, probit, and Poisson GLMs.
- `pf.quantreg(...)` for linear quantile regression.

Read the [getting-started guide](../../../pyfixest/docs/pages/getting-started.md)
before composing an unfamiliar call. Use the
[OLS and fixed-effects tutorial](../../../pyfixest/docs/pages/tutorials/ols-fixed-effects.md),
[Poisson and GLM tutorial](../../../pyfixest/docs/pages/tutorials/poisson-glm.md),
or [quantile-regression tutorial](../../../pyfixest/docs/pages/tutorials/quantile-regression.md)
for estimator-specific behavior.

## Results

An estimation returns a result object such as `Feols`, `Feiv`, `Fepois`, or
`Feglm`. Prefer public methods including `tidy()`, `coef()`, `se()`, `pvalue()`,
`tstat()`, `confint()`, `predict()`, `resid()`, `fixef()`, `vcov()`, and
`summary()`. Check a method's installed signature because support differs by
result type.

Multiple-estimation syntax, `split=`, or `fsplit=` returns `FixestMulti`. Use
`to_list()`, `fetch_model()`, `tidy()`, or reporting functions that accept the
container. Do not reach into private container or model attributes.

`copy_data=True` is the safe default. `store_data=False` and `lean=True` reduce
model state but disable post-estimation operations that need the original data.
