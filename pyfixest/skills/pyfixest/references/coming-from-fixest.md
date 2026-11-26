# Coming from R fixest

pyfixest follows R `fixest` (checked against `fixest` 0.14) where Python
allows, but several spellings differ and some features are missing. Search
this file for the R spelling. Numerical comparisons:
https://pyfixest.org/explanation/compare-fixest-pyfixest.html

```python
import pyfixest as pf

data = pf.get_data()
fit = pf.feols("Y ~ X1 + X2 | f1", data, vcov={"CRV1": "f1"})
```

## Formulas

| R `fixest` | pyfixest |
|---|---|
| `x^2`, `I(x^2)` | `I(x**2)`; `^` is not a power |
| `factor(x)`, `as.factor(x)` | `C(x)` or `i(x)` |
| `i(f, ref = 1)` | `i(f, ref=1)` |
| `i(f, k)`, numeric `k` | `i(f, k)`: one `k` slope per level of `f` |
| `i(f, i.k)`, `i(f, factor(k))` | `i(f, C(k))` |
| `i(f, keep = ...)` | no `keep`; filter rows or merge levels with `bin=` |
| `\| f1^f2` | `\| f1:f2` (`^` still runs, with a `DeprecationWarning`) |
| `\| f1[x]`, `\| f1[[x]]` | same |
| `y ~ x \| f \| d ~ z` | `y ~ x + [d ~ z] \| f` |
| `y ~ x \| d ~ z` | `y ~ x + [d ~ z]` |
| `\| d1 + d2 ~ z1 + z2` | not supported: one endogenous regressor per model |
| `c(y1, y2) ~ x` | `sw(y1, y2) ~ x` |
| `sw()`, `csw()`, `sw0()`, `csw0()`, `mvsw()` | same |
| `..ctrl`, `setFixest_fml()`, `xpd()`, `.[x]`, `..("^x")` | build the string in Python |
| `l(x, 1)`, `f(x)`, `d(x)`, `panel.id = ~id + t` | compute lags in pandas, see below |
| `poly(x, 2)`, `log(x)` | available; `np.` functions also work |
| `sunab(cohort, period)` | `pf.event_study(..., estimator="saturated")` |

```python
data = data.sort_values(["f1", "f2"])
data["X2_lag"] = data.groupby("f1")["X2"].shift(1)
data["X2_diff"] = data["X2"] - data["X2_lag"]
pf.feols("Y ~ X2_lag + X2_diff | f1", data)

controls = ["X2", "Z2"]
pf.feols(f"Y ~ X1 + {' + '.join(controls)} | f1", data)  # replaces ..ctrl
```

## Estimation arguments

| R `fixest` | pyfixest |
|---|---|
| `feglm(family = binomial)` | `pf.feglm(..., family="logit")` |
| `feglm(family = binomial("probit"))` | `family="probit"` |
| `feglm(family = gaussian)`, `poisson` | `family="gaussian"`, `"poisson"`, or `pf.fepois()` |
| `femlm()`, `fenegbin()`, `feNmlm()` | not available |
| `weights = ~w` | `weights="w"` (`weights_type="fweights"` for frequency weights) |
| `offset = ~log(e)` | `offset="log(e)"` (`fepois`, and `feglm` with `family="poisson"`) |
| `split = ~k`, `fsplit = ~k` | `split="k"`, `fsplit="k"` |
| `subset = ~x > 0` | filter the DataFrame first |
| `fixef.rm = "singleton"` | `fixef_rm="singleton"` (default) or `"none"` |
| `fixef.tol`, `fixef.iter` | `demeaner=pf.MapDemeaner(fixef_tol=1e-8, fixef_maxiter=10_000)` |
| `collin.tol`, `lean` | `collin_tol`, `lean` |
| `nthreads` | no argument |

## Standard errors

| R `fixest` | pyfixest |
|---|---|
| `cluster = ~f1`, `vcov = ~f1` | `vcov={"CRV1": "f1"}` |
| `vcov = "cluster"` | name the column: `vcov={"CRV1": "f1"}` |
| `vcov = "twoway"`, `~f1 + f2` | `vcov={"CRV1": "f1 + f2"}` |
| `vcov = ~f1^f2` | add the interaction as a column and cluster on it |
| `se = "hetero"`, `vcov = "hetero"` | `vcov="hetero"` |
| `vcov = NW(2) ~ id + t`, `panel.id` | `vcov="NW", vcov_kwargs={"panel_id": "id", "time_id": "t", "lag": 2}` |
| `vcov = DK(2) ~ t`, `vcov = "DK"` | `vcov="DK"` with the same `vcov_kwargs`; `panel_id` is required |
| `vcov = "conley"` | not available |
| `ssc(K.adj, K.fixef, G.adj, G.df)` | `pf.ssc(k_adj=, k_fixef=, G_adj=, G_df=)` |
| `K.fixef = "nested"` | `k_fixef="nonnested"` (the default in both) |
| `ssc(t.df = ...)` | no argument |
| `ssc = ssc(adj = FALSE)` (old names) | `ssc=pf.ssc(k_adj=False)` |
| `summary(fit, vcov = ...)` | `fit.vcov(...).summary()` |

Both packages default to iid standard errors and drop singleton
fixed-effect groups. Details: `inference.md` in this folder.

## Results

```python
fit.coef(), fit.se(), fit.tstat(), fit.pvalue()  # coef(), se(), tstat(), pvalue()
fit.confint(alpha=0.10)  # confint(fit, level = 0.9)
fit.variance_covariance.vcov  # vcov(fit), an ndarray
fit.sample_info.n_obs  # nobs(fit)
fit.fitstat.r2, fit.fitstat.r2_within  # r2(fit, "r2"), r2(fit, "wr2")
fit.resid(), fit.predict()  # resid(fit), fitted(fit)
fit.predict(newdata=data.head())  # predict(fit, newdata)
fe = fit.fixef()  # fixef(fit): one DataFrame, not a list
fe[fe["variable"] == "f1"]
fit.collinearity.dropped_coef_names  # fit$collin.var

iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)
iv.first_stage.model.summary()  # summary(fit, stage = 1)
iv.first_stage.diagnostics.f_stat  # fitstat(fit, ~ivf)

multi = pf.feols("sw(Y, Y2) ~ X1", data)
multi.fetch_model(0, print_fml=False)  # est[[1]]; multi[0] fails
multi.all_fitted_models["Y2 ~ 1 + X1"]  # est[lhs = "Y2"]; keys are canonical formulas
```

## Tables and plots

| R `fixest` | pyfixest |
|---|---|
| `etable(f1, f2)` | `pf.etable([f1, f2])`: one list; the second positional argument is `type` |
| `etable(..., tex = TRUE)` | `type="tex"` (also `"md"`, `"df"`, `"html"`, `"typst"`, `"docx"`, `"gt"`) |
| `dict = c(x1 = "Label")` | `labels={"x1": "Label"}` |
| `headers = ...` | `model_heads=[...]` |
| `keep`, `drop` (regex) | `keep=[...]`, `drop=[...]` (regex) |
| `file = "t.tex"` | `file_name="t.tex"` |
| `coefplot(fit)`, `iplot(fit)` | `pf.coefplot([fit])`, `pf.iplot(fit)` |

## Behaviour that differs

- **Clustering on interactions** (`~f1^f2`) raises `ValueError`; create the
  column.
- **`i(f, k, ref2=...)` with a numeric `k`** raises `ValueError` instead of
  ignoring `ref2`; write `i(f, C(k), ref2=...)`.
- **`i(a) * x`**: `x` is the slope of the reference level of `a`; `fixest`
  reports the last level's slope. Fitted values agree.
- **Level order of strings** is by code point, so `"Banana"` sorts before
  `"apple"`; R sorts by locale. Set `ref=` explicitly to match.
- **`i()` without an intercept** (`0 + i(f, x) + i(f, g)`) keeps every cell
  of later `i()` terms; `fixest` drops one.
- **Categorical IV instruments without an intercept**, as in
  `Y ~ 0 + X2 + [X1 ~ i(f)]`, keep every level; `fixest` drops a reference
  level.
- **`i()` with `split=`**: levels and `ref` are checked within each split
  sample; a `ref` missing from one sample raises `ValueError`.
- **Fixed-effect columns** need a numeric, `"string"` or `"category"` dtype
  and identifier names; `predict()` needs the fitted dtype.
- **Stepwise operators** combine with other operators: `csw(x1, x2):z` is
  allowed.
- **Indefinite variance-covariance matrices** are not repaired (no `vcov_fix`), so a
  negative variance gives a `NaN` standard error.
- **`feglm(family="gaussian")`** matches `feols()` exactly, including the
  small-sample correction.
- **Multiple estimation names** are canonical formulas (`Y ~ 1 + X1`); print
  `list(multi.all_fitted_models)` before indexing.
