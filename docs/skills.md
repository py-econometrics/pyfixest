# Skills

This page provides a ready-to-use skill file for analytics projects that use PyFixest. The goal is to make LLM-assisted analysis more reliable by giving the model a concise, authoritative reference for PyFixest usage.

## How To Use

1. Copy the skill content below into a file named `SKILL.md` or `Agent.md` (or a tool-specific skill file) in your analytics project.
2. Ensure your LLM tool is configured to read project skills.
3. Use the formula and inference guidelines below as the authoritative PyFixest reference for the model.

## Skill File (PyFixest)

```markdown
# PyFixest Skill

## Core API (4 functions)

- `pyfixest.feols(fml, data, vcov, weights, ssc, fixef_rm, ...)`: OLS/WLS/IV with fixed effects.
- `pyfixest.fepois(fml, data, vcov, ...)`: Poisson regression with fixed effects.
- `pyfixest.feglm(fml, data, family, vcov, ...)`: GLM regression (family: "logit", "probit", "gaussian") with fixed effects.
- `pyfixest.quantreg(fml, data, quantile, ...)`: Quantile regression via interior point solver.

## Formula Syntax

Formulas follow fixest syntax and are split into 1–3 parts by `|`:

- One-part: `"Y ~ X1 + X2"` (no fixed effects, no IV)
- Two-part: `"Y ~ X1 + X2 | FE1 + FE2"` (fixed effects)
- Two-part IV: `"Y ~ X1 + X2 | X_endog ~ Z1 + Z2"` (IV without fixed effects)
- Three-part IV: `"Y ~ X1 + X2 | FE1 + FE2 | X_endog ~ Z1 + Z2"` (IV with fixed effects)

IV behavior:
- The IV part must be `endogenous ~ instruments`.
- Exogenous variables from the second-stage RHS are automatically added to the first stage.
- Endogenous variables are automatically added to the second stage.
- Multiple endogenous variables are not supported.

Other syntax:
- Multiple depvars are expanded to multiple estimations: `"Y1 + Y2 ~ X1"` behaves like `"sw(Y1, Y2) ~ X1"`.
- `i()` creates indicator expansions and interactions:
- `i(cat)` expands to dummies for each level of `cat` (one omitted).
- `i(cat, ref="Base")` sets the omitted reference level explicitly.
- `i(cat, x)` interacts a categorical `cat` with a numeric `x` (varying slopes by category).
- `i(cat1, cat2, ref2="Base")` interacts two categorical variables; `ref2` sets the omitted level of `cat2`.
- Example (cat × numeric): `Y ~ i(industry, exposure)` creates industry-specific slopes on `exposure`.
- Example (cat × cat): `Y ~ i(state, year, ref2=2000)` creates state-by-year indicators with 2000 as the base year.
- Standard interactions still work:
- `X1 * X2` expands to `X1 + X2 + X1:X2`.
- `X1:X2` is the interaction term only (no main effects).
- Interacted FEs: `"Y ~ X1 | FE1 ^ FE2"` (creates a combined FE).

### Multiple Estimation Operators

Operators can appear anywhere in the formula (RHS, fixed effects, IV parts). They can be combined; expansion is recursive and produces all combinations. Note that multiple estimation can be significantly faster than independent model calls due to internal optimisations. 

`sw` (sequential stepwise):
- `y ~ x1 + sw(x2, x3)` → `y ~ x1 + x2` and `y ~ x1 + x3`

`sw0` (sequential stepwise with zero step):
- `y ~ x1 + sw0(x2, x3)` → `y ~ x1`, `y ~ x1 + x2`, `y ~ x1 + x3`

`csw` (cumulative stepwise):
- `y ~ x1 + csw(x2, x3)` → `y ~ x1 + x2`, `y ~ x1 + x2 + x3`

`csw0` (cumulative stepwise with zero step):
- `y ~ x1 + csw0(x2, x3)` → `y ~ x1`, `y ~ x1 + x2`, `y ~ x1 + x2 + x3`

`mvsw` (multiverse stepwise):
- `y ~ mvsw(x1, x2, x3)` → all non-empty combinations plus the zero step:
  `y ~ 1`, `y ~ x1`, `y ~ x2`, `y ~ x3`, `y ~ x1 + x2`, `y ~ x1 + x3`, `y ~ x2 + x3`, `y ~ x1 + x2 + x3`

Combining operators example:
- `y ~ csw(x1, x2) + sw(z1, z2)` expands to:
  `y ~ x1 + z1`, `y ~ x1 + z2`, `y ~ x1 + x2 + z1`, `y ~ x1 + x2 + z2`

You can run regressions for subsamples by using the `split` and `fsplit` arguments, where both split by the 
provided variable, but `fsplit` also provides a fit for the full sample. 

## Inference (vcov)

Pass to `vcov`:

- `"iid"` — IID errors
- `"hetero"` — HC1 heteroskedasticity-robust (alias: `"HC1"`, `"HC2"`, `"HC3"`)
- `{"CRV1": "cluster_var"}` — Cluster-robust variance
- `{"CRV3": "cluster_var"}` — Leave-one-cluster-out jackknife
- `"nw"` — Newey-West HAC (requires panel_id and time_id)
- `"dk"` — Driscoll-Kraay HAC (requires panel_id and time_id)

## Post Processing

Model objects support:

- `.summary()` — Print regression summary
- `.tidy()` — Tidy DataFrame of coefficients, SEs, t-stats, p-values, CIs
- `.coef()` — Coefficient values
- `.se()` — Standard errors
- `.pvalue()` — P-values
- `.confint()` — Confidence intervals
- `.predict(newdata)` — Predictions
- `.resid()` — Residuals
- `.vcov()` — Variance-covariance matrix
- `.wildboottest(param, reps, seed)` — Wild cluster bootstrap inference
- `.ccv(treatment, pk, qk, ...)` — Causal cluster variance estimator
- `.ritest(param, reps, ...)` — Randomization inference
- `.decompose(param, x1_vars, type, ...)` — Gelbach (2016) decomposition
```
