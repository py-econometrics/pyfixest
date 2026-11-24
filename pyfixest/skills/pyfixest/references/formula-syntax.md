# Formula syntax reference

Every formula has the shape
`"dependent ~ regressors + [endogenous ~ instruments] | fixed effects"`. The
right-hand side of the main equation follows
[formulaic](https://matthew.wardrop.casa/formulaic/) grammar plus the pyfixest
operators below. Every example runs as written;
later blocks reuse `pf` and `data`. Full tutorial:
https://pyfixest.org/tutorials/formula-syntax.html

```python
import numpy as np
import pyfixest as pf

data = pf.get_data()  # Y, Y2, X1, X2, Z1, Z2 numeric; f1, f2, f3, group_id float codes
```

## Regressors, intercept and transforms {#covariates-intercept-and-transforms}

```python
formulas = [
    "Y ~ X1 + X2",  # intercept added automatically
    "Y ~ 0 + X1",  # no intercept; "Y ~ -1 + X1" and "Y ~ X1 - 1" too
    "Y ~ X1:X2",  # product only
    "Y ~ X1 * X2",  # X1 + X2 + X1:X2
    "Y ~ X1 + I(X1**2)",  # square: Python arithmetic inside I()
    "Y ~ X1 + I(X1 * X2)",  # any Python expression inside I()
    "Y ~ poly(X1, 2)",  # orthogonal polynomial, columns poly(X1, 2)[1], [2]
    "Y ~ np.exp(X2) + np.square(X2)",  # numpy is available as np
    "Y ~ log(X2 + 10)",  # log() is np.log
]
for fml in formulas:
    pf.feols(fml, data)
```

- Powers: write `I(X1**2)`. `^` is not a power operator, neither in the
  formula nor inside `I()`.
- `log()` of zero gives `-inf`; such rows are dropped with a warning and
  counted in `fit.sample_info.dropped_by_stage.infinite`.
- Rows with missing values in any model variable are dropped
  (`dropped_by_stage.missing`).
- Column names with spaces or dots go in backticks: ``"Y ~ `my var`"``. Fixed
  effects cannot be quoted this way; rename those columns.
- Your own functions need `context=0` (the calling scope) or a dict:

```python
def winsorize(x):
    return np.clip(x, -2, 2)

pf.feols("Y ~ winsorize(X2)", data, context=0)
pf.feols("Y ~ winsorize(X2)", data, context={"winsorize": winsorize})
```

## Categorical variables: `C()` and `i()`

`C(f3)` makes dummies with formulaic names such as `C(f3)[T.1.0]`; the first
level is the reference when the model has an intercept. `i()` makes dummies
named `f3::1.0` and takes these keyword arguments:

| Argument | Meaning |
|---|---|
| `ref=` | reference level of the first variable to leave out |
| `ref2=` | reference level of a categorical second variable |
| `bin=` | dict `{new_level: [old levels]}` merging levels of the first variable |
| `bin2=` | the same for a categorical second variable |

```python
pf.feols("Y ~ i(f3)", data)  # intercept present: first level (0.0) left out
pf.feols("Y ~ 0 + i(f3)", data)  # no intercept: every level kept
pf.feols("Y ~ i(f3, ref=5)", data)  # reference level 5.0 (5 matches the float code)
pf.feols("Y ~ i(f3, bin={'low': [0, 1, 2]}, ref='low')", data)  # merge, then use as reference
pf.feols("Y ~ i(f3, X2)", data)  # one X2 slope per f3 level: f3::0.0:X2, ...
pf.feols("Y ~ i(f3, X2, ref=0)", data)  # slopes for every level except 0
pf.feols("Y ~ i(f3, C(f2), ref=0, ref2=1)", data)  # f3-by-f2 dummies: f3::1.0:f2::0.0, ...
```

- The second argument decides the meaning: a numeric or boolean variable
  gives one slope per level of the first; wrap it in `C()` to interact two
  factors. A string or categorical second variable is a factor already.
- `ref2=` and `bin2=` with a numeric second variable not wrapped in `C()`
  raise `ValueError`.
- A `ref` that is not a level of the variable raises `ValueError`.
- `i(f3) * X2` follows formulaic: `X2` is the slope of the reference level and
  `f3::k:X2` are differences from it. `i(f3) + i(f3, X2)` gives one slope per
  level instead.
- `keep=` is not an `i()` argument; filter rows or merge levels with `bin=`.
- Event studies: `i(rel_year, treated, ref=-1)` interacts relative time with
  a treatment dummy; `pf.iplot(fit)` plots the `i()` coefficients.

## Fixed effects

```python
pf.feols("Y ~ X1 | f1", data)
pf.feols("Y ~ X1 | f1 + f2 + f3", data)  # any number of fixed effects
pf.feols("Y ~ X1 | f1:f2", data)  # one effect per f1-by-f2 cell
pf.feols("Y ~ X1 | f1 + f1:f2", data)
```

- A fixed effect must be a data column or a `:` interaction of columns.
  Transforms such as `C(f1)`, `factor(f1)` or `I(f1 + f2)` raise
  `FormulaSyntaxError`; create the column first.
- Names must be valid Python identifiers (rename `firm.id` to `firm_id`).
- Column dtype must be numeric, `"string"` or `"category"`, not `object`:
  convert with `df["firm"] = df["firm"].astype("string")`.
  `predict(newdata=...)` needs the same dtype as the fitting data.
- Rows in singleton levels are dropped with a warning
  (`fixef_rm="singleton"`, the default); `fixef_rm="none"` keeps them.
- Fixed-effect estimates are not coefficients; recover them with
  `fit.fixef()` (a DataFrame).

## Varying slopes

```python
pf.feols("Y ~ X1 | f1[X2]", data)  # f1 effects plus an X2 slope per f1 level
pf.feols("Y ~ X1 | f1[[X2]]", data)  # the slopes only; Intercept is reported
pf.feols("Y ~ 0 + X1 | f1[[X2]]", data)  # slopes only, no intercept
pf.feols("Y ~ X1 | f1[X2, Z2] + f2", data)  # several slopes, plus other effects
pf.feols("Y ~ X1 | f1:f2[X2]", data)  # slopes on an interacted group
```

Slopes are absorbed like fixed effects and not reported. They work in
`feols()` (also IV), `fepois()` and `feglm()` with the default demeaner.
`predict()`, `fixef()`, `wildboottest()`, `ritest()`, `decompose()`, CRV3 and
HAC standard errors do not support them; `fit.capabilities` says which work.

## Instrumental variables

```python
pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)  # X1 endogenous, Z1 instrument
pf.feols("Y ~ [X1 ~ Z1 + Z2]", data)  # over-identified
pf.feols("Y ~ 0 + X2 + [X1 ~ Z1]", data)  # no intercept in either stage
pf.feols("Y ~ X2 + [X1:X2 ~ Z1:X2]", data)  # interacted endogenous term
```

- One endogenous regressor per model; more raise `FormulaSyntaxError`.
- Only `feols()` fits IV.
- The main formula decides the intercept of both stages; a `0` or `1` inside
  the brackets has no effect.
- Fixed effects go after `|` and apply to both stages.
- The first stage is `fit.first_stage.model`; diagnostics are in
  `fit.first_stage.diagnostics` (`f_stat`, `p_value`, and `eff_f` after
  `fit.IV_Diag()`).

## Multiple estimation

One formula can describe many models. pyfixest fits all of them and returns a
`FixestMulti`; models with the same fixed effects and sample reuse demeaned
columns, so this is faster than a loop.

| Operator | `Y ~ X1 + op(X2, Z1)` fits |
|---|---|
| `sw(X2, Z1)` | `X1 + X2`; `X1 + Z1` |
| `sw0(X2, Z1)` | `X1`; `X1 + X2`; `X1 + Z1` |
| `csw(X2, Z1)` | `X1 + X2`; `X1 + X2 + Z1` |
| `csw0(X2, Z1)` | `X1`; `X1 + X2`; `X1 + X2 + Z1` |
| `mvsw(X2, Z1)` | `X1`; `X1 + X2`; `X1 + Z1`; `X1 + X2 + Z1` |

```python
multi = pf.feols("sw(Y, Y2) ~ X1 | csw0(f1, f2)", data)  # outcomes and fixed effects
print(list(multi.all_fitted_models))  # 'Y ~ 1 + X1', 'Y ~ X1 | f1', ...
pf.feols("Y ~ csw(X1, X2) + sw(Z1, Z2)", data)  # every combination: 4 models
pf.feols("Y ~ csw(X1, X2):Z1", data)  # X1:Z1; X1:Z1 + X2:Z1 (each step grouped)
pf.feols("Y ~ X1 + sw0(X2) | f1", data)  # one argument: with and without X2
pf.feols("Y ~ X2 + [X1 ~ sw(Z1, Z2)]", data)  # operators work inside the IV block
pf.feols("Y ~ X1 | f1", data, split="f3")  # one model per value of f3
pf.feols("Y ~ X1 | f1", data, fsplit="f3")  # full sample first, then each value
```

- Operators work among regressors, in fixed effects and inside the instrument block. On
  the left-hand side only `sw` works (`sw(Y, Y2) ~ X1`); `csw`, `csw0` and
  `mvsw` there raise `FormulaSyntaxError`. Several operators combine as a
  Cartesian product.
- An expansion that mixes IV and non-IV models, such as
  `Y ~ X1 + sw([X2 ~ Z1], Z2)`, raises `FormulaSyntaxError`; fit them
  separately.
- Expansions that produce the same model twice fit it once, with a warning.
- Model keys are canonical formulas: no fixed effects gives `Y ~ 1 + X1`,
  split samples end in `(Sample: f3 = 1.0)`, and `fsplit` adds
  `(Sample: f3 = all)`. Print `list(multi.all_fitted_models)` before
  indexing.
- `split=` and `fsplit=` take one column name (no `~`).

## Spellings that changed in 0.70

These still run with a `DeprecationWarning`:

```text
Y ~ X1 | f1^f2          ->  Y ~ X1 | f1:f2
Y ~ X1 | f1 | X2 ~ Z1   ->  Y ~ X1 + [X2 ~ Z1] | f1
Y + Y2 ~ X1             ->  sw(Y, Y2) ~ X1
```

R-only spellings such as `factor(x)`, `c(Y, Y2) ~`, `l(x, 1)` and `..ctrl`
fail; see `coming-from-fixest.md` in this folder.
