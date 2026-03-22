## Setup

```{python}
import numpy as np
import pyfixest as pf
data = pf.get_data()
data.head()
```

`PyFixest` specifies different regression models by Wilkinson Formulas via the `formulaic` package. Wilkinson formulas should be familiar to you if you have used R's `lm()` or `statsmodels` formula API. Many additional ideas implemented in `PyFixest`  have been developed in the `fixest` package (most notably multiple estimation syntax, the i-operator, sample splitting). By default, all formula options presented here are supported by all models available via the `pf.feols()`, `pf.feglm()`, and `pf.fepois()` APIs.

## Basic Syntax

In the simplest case, we regress covariates `X1` and `X2` on `Y`.

```{python}
fit1 = pf.feols("Y ~ X1 + X2", data=data)
fit1.summary()
```

All transformations that are supported via `formulaic` are also supported via `PyFixest`. To name just a few important ones,
you can create categorical variables via the `C()` operator:

```{python}
fit2 = pf.feols("Y ~ X1 + X2 + C(f1)", data=data)
```

You can interact variables via the `*` and `:` operators:

```{python}
fit3 = pf.feols("Y ~ X1:X2", data=data)
fit4 = pf.feols("Y ~ X1*X2", data=data)
pf.etable([fit3, fit4])
```

To create logarithms of a function, just use

```{python}
fit5 = pf.feols("Y ~ log(X1)", data=data)
```

or use any `numpy` transforms, e.g.

```{python}
fit5 = pf.feols("Y ~ X1  + np.power(X1,2)", data=data)
```

Note - for the logarithm, we suggest to not rely on `np.log` but use the internal `log` operator.

## Fixed Effects Syntax

We can add fixed effects behind the `|` operator: here we add two fixed effects `f1` and `f2`.

```{python}
fit6 = pf.feols("Y ~ X1 + X2 | f1 + f2", data=data)
```

We can interact two fixed effects via the `^` operator.

```{python}
fit7 = pf.feols("Y ~ X1 + X2 | f1^f2", data=data)
```

For details on fixed effects regression, take a look at the [OLS with Fixed Effects](ols-fixed-effects.qmd) vignette.

## Instrumental Variables (IV) Syntax

For IV estimation, `PyFixest` uses a three-part formula syntax:

```python
"Y ~ exogenous_controls | fixed_effects | endogenous ~ instruments"
```

Here is a minimal example with fixed effects:

```{python}
fit_iv = pf.feols("Y ~ X2 | f1 + f2 | X1 ~ Z1", data=data)
fit_iv.summary()
```

For details on IV estimation, take a look at the
[Instrumental Variables](instrumental-variables.qmd) vignette.

## The `i()` operator for interacting fixed effects

For interacting fixed effects, we include a specialised operator `i()`

If you simply wrap a variable into `i()`, it will be treated just as the `C()` operator (see above).

```{python}
fit_i = pf.feols("Y ~ i(f1)", data=data)
fit_c = pf.feols("Y ~ C(f1)", data=data)
```

But overall, `i()` is more powerful than `C()`. Most importantly, you can easily set the reference
level of the categorical variable:

```{python}
# set 1 as reference level
fit_i1 = pf.feols("Y ~ i(f1, ref = 1)", data=data)
```

You can also easily interact variables:

```{python}
# set 1 as reference level
fit_i2 = pf.feols("Y ~ i(f1, f2)", data=data)
```

and set reference levels for both via the `ref` and `ref2` levels.

```{python}
# set 1 as reference level
fit_i3 = pf.feols("Y ~ i(f1, f2, ref = 1, ref2 = 2)", data=data)
```

This is in particular useful for difference-in-differences models.

Last, you can bin levels of a variable via the `bin` argument. This groups multiple levels into a single category.

```{python}
fit_bin = pf.feols(
    "Y ~ i(f1, bin={'low': list(range(0, 10)), 'mid': list(range(10, 20)), 'high': list(range(20, 30))}, ref='low')",
    data=data,
)
fit_bin.summary()
```

## Multiple Estimation Syntax

Last, `PyFixest` provides syntactic sugar to fit multiple estimations in one go. This is not only economizes on lines-of-code, but allows for performance optimizations via caching - if you fit many regression models on a fixed set of fixed effects and many overlapping covariates or dependent variables, and performance is poor, we highly recommend you to try out multiple estimations.

For multiple estimations, we provide 5 custom operators: `sw`, `csw`, `sw0`, `csw0` and `mvsw`. In addition, it is possible to specify multiple dependent variables.

### Multiple dependent variables

Multiple depvars are expanded to multiple estimations: `"Y1 + Y2 ~ X1"` behaves like `"sw(Y1, Y2) ~ X1"`.

```{python}
fit_multi_dep = pf.feols("Y + Y2 ~ X1 + X2", data=data)
pf.etable(fit_multi_dep)
```

### `sw()`: stepwise alternatives

`y ~ x1 + sw(x2, x3)` expands to `y ~ x1 + x2` and `y ~ x1 + x3`.

```{python}
fit_sw = pf.feols("Y ~ X1 + sw(X2, Z1)", data=data)
pf.etable(fit_sw)
```

### `sw0()`: stepwise with zero step

`y ~ x1 + sw0(x2, x3)` expands to `y ~ x1`, `y ~ x1 + x2`, and `y ~ x1 + x3`.

```{python}
fit_sw0 = pf.feols("Y ~ X1 + sw0(X2, Z1)", data=data)
pf.etable(fit_sw0)
```

### `csw()`: cumulative stepwise

`y ~ x1 + csw(x2, x3)` expands to `y ~ x1 + x2` and `y ~ x1 + x2 + x3`.

```{python}
fit_csw = pf.feols("Y ~ X1 + csw(X2, Z1)", data=data)
pf.etable(fit_csw)
```

### `csw0()`: cumulative stepwise with zero step

`y ~ x1 + csw0(x2, x3)` expands to `y ~ x1`, `y ~ x1 + x2`, and `y ~ x1 + x2 + x3`.

```{python}
fit_csw0 = pf.feols("Y ~ X1 + csw0(X2, Z1)", data=data)
pf.etable(fit_csw0)
```

### `mvsw()`: multiverse stepwise

`y ~ mvsw(x1, x2, x3)` expands to all non-empty combinations plus the zero step: `y ~ 1`, `y ~ x1`, `y ~ x2`, `y ~ x3`, `y ~ x1 + x2`, `y ~ x1 + x3`, `y ~ x2 + x3`, `y ~ x1 + x2 + x3`.

```{python}
fit_mvsw = pf.feols("Y ~ mvsw(X1, X2, Z1)", data=data)
pf.etable(fit_mvsw)
```

### Combining operators

Multiple estimation operators can be combined. For example, `y ~ csw(x1, x2) + sw(z1, z2)` expands to `y ~ x1 + z1`, `y ~ x1 + z2`, `y ~ x1 + x2 + z1`, `y ~ x1 + x2 + z2`.

```{python}
fit_combo = pf.feols("Y ~ csw(X1, X2) + sw(Z1, X1:Z1)", data=data)
pf.etable(fit_combo)
```

## Regressions on Multiple Samples

Via the `split` and `fsplit` argument, you can easily separate identical models on different samples.

- `split` estimates separate models by subgroup.
- `fsplit` does the same but also keeps the full-sample fit.

```{python}
fit_split = pf.feols("Y ~ X1 + X2 | f1", data=data, split="f2")
pf.etable(fit_split)
```

```{python}
fit_fsplit = pf.feols("Y ~ X1 + X2 | f1", data=data, fsplit="f2")
pf.etable(fit_fsplit)
```

## Where to Go Next

- [OLS with Fixed Effects](ols-fixed-effects.qmd): practical FE estimation patterns.
- [Difference-in-Differences](difference-in-differences.qmd): event-study applications with `i()`.
- [Regression Tables](regression-tables.qmd): organize and export many-model workflows.
- [How-To: Translating Stata to PyFixest](../how-to/stata-2-pyfixest.qmd): syntax mapping and defaults.
