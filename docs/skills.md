---
title: "PyFixest skill for AI agents"
description: "A version-controlled skill for using PyFixest reliably."
---

# PyFixest skill for AI agents

This page is generated from the canonical files in `skills/pyfixest`.
Copy the directory as a unit so its focused references remain available.

## `SKILL.md`

````markdown
---
name: pyfixest
description: "Use for fixest-style econometrics in Python with PyFixest: formulas, OLS/WLS/IV, high-dimensional fixed effects, GLMs, quantile regression, inference, reporting, DiD, demeaning backends, or troubleshooting. Read the bundled version-matched docs before guessing syntax, covariance options, or result APIs."
---

# PyFixest

Use the documentation installed with the package before relying on memory. Start
at `pyfixest/docs/index.md`; its content matches the installed PyFixest version.
Use <https://pyfixest.org/llms.txt> when the local corpus is unavailable.

## Workflow

1. Identify the estimator, formula features, inference method, and output the
   user needs.
2. Read the relevant reference below and follow its links into the bundled
   documentation.
3. Check the installed function signature or API reference for the installed
   version before writing code.
4. Prefer public `pyfixest` functions and result methods. Do not depend on
   underscore-prefixed model state.
5. State unsupported estimator/inference combinations instead of silently
   substituting a method.
6. Use small, reproducible examples and preserve the user's input data.

## Focused references

- [Core API and result objects](references/core-api.md)
- [Formula syntax and multiple estimation](references/formula-syntax.md)
- [Standard errors and inference](references/inference.md)
- [Tables, plots, and model reporting](references/reporting.md)
- [Specialized estimators and causal workflows](references/specialized-estimators.md)
- [Demeaning backends](references/demeaners.md)
- [Troubleshooting and model state](references/troubleshooting.md)
````

## `references/core-api.md`

````markdown
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
````

## `references/formula-syntax.md`

````markdown
# Formula syntax and multiple estimation

PyFixest combines Formulaic expressions with fixest-style fixed effects and
multiple-estimation operators. Read the bundled
[formula guide](../../../pyfixest/docs/pages/tutorials/formula-syntax.md) before
generating a complex formula.

## Common forms

```python
pf.feols("Y ~ X1 + X2", data=df)
pf.feols("Y ~ X1 + X2 | firm + year", data=df)
pf.feols("Y ~ X1 + [X2 ~ Z1] | firm:year", data=df)
```

- Put fixed effects after `|`.
- Use `:` for interacted fixed effects, such as `firm:year`. The old `^`
  spelling is deprecated.
- Write IV terms in brackets as `[endogenous ~ instruments]`. Legacy three-part
  IV formulas remain accepted during deprecation but should not be generated.
- Use Formulaic operators and transforms for ordinary terms. `X1 * X2` expands
  to main effects and their interaction; `X1:X2` is the interaction only.
- Use `i(variable, ref=...)` for indicators and event-study-style interactions.

## Multiple estimation

`sw`, `sw0`, `csw`, `csw0`, and `mvsw` expand one formula into a model grid.
Multiple dependent variables, `split=`, and `fsplit=` can also produce multiple
models. The result is a `FixestMulti`, not a single fitted model.

Avoid manually expanding a large grid until you have checked whether a native
operator expresses it. Native expansion can reuse work across models.
````

## `references/inference.md`

````markdown
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
````

## `references/reporting.md`

````markdown
# Tables, plots, and model reporting

Use public result methods for data and package-level functions for presentation.

- `fit.tidy()` returns a tidy table of estimates and inference.
- `fit.summary()` or `pf.summary(...)` prints model summaries.
- `pf.etable(...)` builds regression tables from a model, list, or
  `FixestMulti`.
- `pf.coefplot(...)` plots coefficient estimates.
- `pf.iplot(...)` focuses on coefficients created by `i()`.
- `pf.qplot(...)` compares quantile-regression estimates.

Read the bundled
[regression-tables tutorial](../../../pyfixest/docs/pages/tutorials/regression-tables.md)
before selecting `etable` formatting, labels, coefficient layouts, or output
types. Check the installed signatures for plotting backends and optional
dependencies.

Keep analysis separate from rendering: first fit and validate the models, then
pass the result objects to a reporting function. Do not extract private model
state to construct a table.
````

## `references/specialized-estimators.md`

````markdown
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
````

## `references/demeaners.md`

````markdown
# Demeaning backends

The `demeaner=` argument controls how high-dimensional fixed effects are removed.
Read the bundled
[demeaner guide](../../../pyfixest/docs/pages/how-to/demeaner-backends.md) and
[difficult-fixed-effects explanation](../../../pyfixest/docs/pages/explanation/difficult-fixed-effects.md)
before overriding the default.

- `MapDemeaner` is the default and is appropriate for most problems.
- `LsmrDemeaner` is useful for difficult or sparse fixed-effect structures.
- Optional Numba and Torch backends have extra dependency and hardware
  requirements; verify availability before selecting them.

Start with the default, inspect convergence diagnostics or errors, and change
the backend only for a concrete numerical or performance reason. A reusable
preconditioner is valid only when the fixed-effect structure is unchanged.
Treat tolerances, iteration limits, and backend selection as numerical choices
that should be reported when they affect reproducibility.
````

## `references/troubleshooting.md`

````markdown
# Troubleshooting and model state

When a call fails or a result method is unavailable:

1. Confirm the installed PyFixest version and inspect the installed signature.
2. Reduce the example to one estimator, one formula, and a small data sample.
3. Check column names, data types, missing values, formula syntax, and whether
   optional dependencies are installed.
4. Check whether weights, fixed effects, IV, or a multiple-estimation result
   changes feature support.
5. Check whether `store_data=False` or `lean=True` removed state needed by the
   requested post-estimation operation.
6. Preserve the full exception and warnings; do not catch them and guess a
   substitute estimator.

For formula failures, use the bundled
[formula guide](../../../pyfixest/docs/pages/tutorials/formula-syntax.md). For
demeaning or convergence problems, use the
[difficult-fixed-effects explanation](../../../pyfixest/docs/pages/explanation/difficult-fixed-effects.md).
For differences from R `fixest`, consult the
[comparison guide](../../../pyfixest/docs/pages/explanation/compare-fixest-pyfixest.md).

Underscore-prefixed attributes are internal implementation details. If the
public API does not expose a value, consult the installed API documentation or
ask for a supported path rather than depending on private state.
````
