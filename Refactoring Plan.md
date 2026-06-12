# PyFixest Internal Refactoring Plan



Goal: a modular, functional, well-organised internal architecture that is easy to maintain long-term, **without changing the user-facing APIs** (`feols`, `fepois`, `feglm`, `quantreg`, `etable`, post-estimation methods).



Each step below is independently shippable: the full test suite (incl. `test_vs_fixest`) must stay green after every step, so you can merge step by step and release in between.



---



## Current state (scan summary)



### Call flow today (OLS)



```

pf.feols(fml, data, ...) # api/feols.py (~600 lines, mostly docstring)

├─ _resolve_demeaner() # legacy demeaner_backend/fixef_tol translation + warnings

├─ _estimation_input_checks()

└─ FixestMulti(...) # god class

├─ _prepare_estimation() # Formula.parse_to_dict, sets ~15 attrs

└─ _estimate_all_models() # triple loop: splits × fixef keys × formulas

├─ model_kwargs built via if-chains on string self._method

├─ model_map[(method, is_iv)] → ModelClass(**kwargs)

└─ imperative pipeline: prepare_model_matrix → get_fit → vcov →

get_inference → get_performance → wald_test → _clear_attributes

└─ return fetch_model(0) or FixestMulti

```



### Demeaner call chain: 6 Python layers



1. `api/feols.py` → `_resolve_demeaner` (internals/demeaner_options.py)

2. `FixestMulti._estimate_all_models` → passes `demeaner` into model kwargs

3. `Feols.demean()` (models/feols_.py)

4. `demean_model()` (internals/demean_.py) — caching via `lookup_demeaned_data`

5. `dispatch_demean()` (internals/demean_.py) — backend dispatch, preconditioner resolution

6. `demean()` / `demean_within()` (core/demean.py) — dtype casts, single-FE MAP fallback

7. Rust: `_demean_rs` / `_demean_within_rs`



Inconsistency: `Fepois.get_fit` and `Feglm.residualize` skip layer 4 and call

`dispatch_demean` directly, so IWLS paths have their own demeaning conventions

and no cache.



### Other observations



- `Feols` is 2,670 lines: estimation + 6 vcov estimators + wald + wildboottest +

ccv + decompose + fixef + predict + ritest + plotting + update.

- The fit pipeline mutates attributes in place; `to_array()` *overwrites*

`self._Y`/`self._X` (DataFrames) with demeaned numpy arrays — same name, different

meaning at different pipeline stages. Hard to reason about and the main blocker

for a numpy-style API.

- `Feols.__init__` requires `FixestFormula` + a DataFrame, so no array entry point exists.

- `FixestMulti` mixes four jobs: formula expansion, model construction (registry),

fit orchestration, and results presentation (tidy/coef/etable bindings).

- Deprecated/legacy surface (all already warning at runtime):

- `demeaner_backend`, `fixef_tol`, `fixef_maxiter` args + `_LSMR_PRESETS` string table (incl. `rust-cg` alias)

- `jax` MAP backend (`estimation/jax/`), `solver="jax"`

- `cupy`/`scipy` LSMR backends (`estimation/cupy/`)

- `use_compression` + `FeolsCompressed` (+ `reps`/`seed` plumbing for it)

- `estimation/deprecated/` (old `FormulaParser`, old `model_matrix_fixest` — the latter still exported from `pyfixest.estimation.__init__`!)

- Back-compat shims `estimation/feols_.py`, `fepois_.py`, `feiv_.py`

- Old `ssc` kwarg names (`adj`, `fixef_k`, `cluster_df`, `cluster_adj`)

- Deprecated relabel helpers in `report/utils.py`



---



## Step 0 — Safety net (before touching anything)



1. **Public-API snapshot test**: assert the exact signature set of `pf.feols/fepois/feglm/quantreg/etable/...` and the public attributes/methods of returned objects. This is the contract the whole refactor must not break.

2. **Golden-results test**: a small battery of models (OLS, WLS, IV, Poisson, logit, multi-estimation, split, CRV/HAC) whose coefficients, SEs, and tidy() output are pinned to stored values. `test_vs_fixest` covers correctness vs R; the golden file makes refactor regressions visible without R.

3. Record coverage baseline; fail CI if coverage of `pyfixest/estimation` drops.

4. Write the deprecation policy into CONTRIBUTING.md (e.g. "warn ≥ 2 minor releases, remove at the next minor"). Everything in Step 1 already warns, so removal is policy-compliant.



Effort: small. Risk: none.



---



## Step 1 — Delete deprecated code



Order within the step is chosen so each sub-step is a small, reviewable PR.



1. **Legacy demeaner args**: remove `demeaner_backend`, `fixef_tol`, `fixef_maxiter` from `feols/fepois/feglm` signatures; delete the legacy branch of `_resolve_demeaner` and `_LSMR_PRESETS`. `_resolve_demeaner` collapses to `demeaner or MapDemeaner()`.

2. **JAX**: delete `estimation/jax/`, the `"jax"` member of `MapBackend`, `solver="jax"` (literals.py, solvers.py), and `_warn_if_deprecated_solver`.

3. **CuPy/SciPy LSMR**: delete `estimation/cupy/`, the `"cupy"` member of `LsmrBackend`, the cupy branch of `dispatch_demean`, cupy-specific validation in `LsmrDemeaner.__post_init__`, and `_warn_if_deprecated_demeaner_backend` (nothing left to warn about).

4. **Compression**: delete `models/feols_compressed_.py`, the `"compression"` method in `FixestMulti`, and `use_compression`/`reps` plumbing (keep `seed` where quantreg uses it). Decide: drop the kwargs outright vs. keep one release raising a clear error. Recommendation: keep `use_compression=False` accepted-but-erroring for one release, then drop.

5. **`estimation/deprecated/`**: delete old `FormulaParser` and old `model_matrix_fixest`. Note: `model_matrix_fixest` is re-exported from `pyfixest.estimation` — technically public. If it has external users, deprecate the export for one release first; otherwise remove now.

6. **Shims**: delete `estimation/feols_.py`, `fepois_.py`, `feiv_.py` root shims.

7. **Misc**: old `ssc` kwarg names in `utils/utils.py`; deprecated relabel helpers in `report/utils.py`.

8. **Decision point — numba MAP backend**: you said "multiple backend options" are slated for deprecation. Recommended end state: `MapDemeaner` (rust, no `backend` field), `LsmrDemeaner(backend="within"|"torch")`. If you agree, add the numba DeprecationWarning now and delete `estimation/numba/` + the `backend` field of `MapDemeaner` in a later release.



Also in this step: delete the corresponding tests, prune `pyproject.toml` extras (`jax`), and update the demeaner-backends vignette.



Payoff: `demeaner_options.py` shrinks from ~180 to ~20 lines, `dispatch_demean` loses two of four branches, two whole subpackages disappear, and every later step touches less code.



Effort: medium (mechanical). Risk: low — everything already warns.



---



## Step 2 — Collapse the demeaning stack (6 layers → 3)



Target call chain:



```

api: demeaner = demeaner or MapDemeaner() # config only, frozen dataclass (as today)

model: demeaner.demean(x, flist, weights, cache=...) # ONE entry point

rust/torch kernel

```



1. Make demeaning a **method on the demeaner config** (strategy pattern):

`MapDemeaner.demean(...)` and `LsmrDemeaner.demean(...)`. This merges today's

`dispatch_demean` + the `core/demean.py` wrappers into the strategy objects.

`core/demean.py` keeps only the thin public `demean()`/`demean_within()`

functions (they are documented public API) implemented via the same strategies.

2. Extract the caching logic of `demean_model` into an explicit **`DemeanCache`**

class (wraps today's `lookup_demeaned_data: dict[frozenset[int], DataFrame]`,

including the "which columns are already demeaned" diffing and preconditioner

reuse). `demean_model` becomes a ~30-line function: check cache → call

`demeaner.demean` → update cache.

3. **Unify the IWLS path**: `Fepois.get_fit` and `Feglm.residualize` call the same

`demeaner.demean(...)` entry point instead of `dispatch_demean`;

`_override_demeaner_tol` becomes `dataclasses.replace` at the call site or a

`demeaner.with_tol(tol)` helper.

4. Move preconditioner seeding/reuse into `DemeanCache` so model classes no longer

carry `_preconditioner` / `_seed_preconditioner` logic.



Resulting layers: model → (cache) → strategy → kernel. The api layer only

constructs config; `FixestMulti` only forwards it.



Effort: medium. Risk: medium — preconditioner caching semantics need the golden tests from step 0 plus the existing `test_within_*` suites.



---



## Step 3 — Extract a pure, functional estimation core



This is the heart of "modular + functional" and the foundation of the numpy API.

Create `pyfixest/estimation/core/` (or extend `internals/`) with **pure functions

on numpy arrays + small frozen dataclasses**, no DataFrames, no self-mutation:



```

fit_ols(X, y, weights, solver) -> OlsFit(beta, residuals, scores, hessian, ...)

fit_iv(X, Z, y, weights, solver) -> IvFit(...)

fit_iwls(X, y, fe, family, demeaner, ...) -> GlmFit(...) # shared Poisson/GLM loop

vcov_iid / vcov_hetero / vcov_hac / vcov_crv1 / vcov_crv3(fit, spec) -> VcovResult

inference(beta, vcov, df) -> Inference(se, tstat, pvalue, conf_int)

```



1. Move the math bodies of `Feols.get_fit`, `Feols._vcov_*`, `get_inference`,

`get_performance` into these functions. The methods on `Feols` become 3-line

delegations — **no behavior change, no API change**.

2. Same for the IWLS loop: today it is duplicated (with variations) in

`Fepois.get_fit` and `Feglm.get_fit`. Extract one parameterized

`fit_iwls(family=...)`; `Fepois` becomes `Feglm` with a Poisson family (fixest

does exactly this). This likely deletes ~300 duplicated lines and is the single

biggest maintainability win in the model layer.

3. Kill the attribute-overwriting pipeline. **(Implemented: 3a and 3b done in
one pass on refactor-demeaner.)** Two distinct smells, fixed in two
phases (do 3a before 3b; 3a is risk-free, 3b changes intermediate arrays):

**3a — Explicit data-stage naming (zero numeric change).**
Today `self._X` means raw DataFrame → demeaned array (`to_array()`) →
sqrt-weighted array (`wls_transform()`) depending on pipeline stage, and
`self._weights` is overwritten with IRLS weights by Fepois/Feglm. Band-aids
like `_X_untransformed` / `_Y_untransformed` exist only because of this.

Naming convention:

| name | type | meaning | mutated? |
|---|---|---|---|
| `Y_df, X_df, Z_df, fe_df` | DataFrame | raw model matrices from formulaic | never |
| `Y_demeaned, X_demeaned` | ndarray | demeaned, collinearity-pruned | never after fit |
| `weights_user` | ndarray | user-supplied weights | never |
| `weights_irls` | ndarray | final IRLS weights (GLM only) | set once |

Implementation: a small `FitState` (or `ModelData`) dataclass passed through
`prepare → demean → fit`, holding one field per stage. No backwards
compatibility needed for `fit._X` / `fit._u_hat` (decided: underscore
attributes are internal; users are pointed to `fit.resid()` etc.) — rename
directly, no aliasing properties. Internal call sites (wildboottest, ccv,
predict, fixef) are updated in the same PR. `to_array()` and the overwrites
at the end of `Fepois/Feglm.get_fit` (`self._Y = WZ`, `self._X = WX`,
`self._weights = combined_weights`) disappear as *state mutations* — the
values live under their stage names.

**3b — Move weights out of the design matrices (representation change,
algebraically equivalent).**
Stop pre-multiplying X/Y/Z by `sqrt(weights)` in `wls_transform`. Instead:
`fit_ols(X, Y, weights)` / `fit_iv(X, Z, Y, weights)` apply weighting
internally (compute X'WX, X'WY); the IWLS cores already form their own
weighted products and need only signature alignment. `vcov_*` take
`weights` explicitly and decide how weights enter — this deletes the
scattered un-weighting hacks (`scores / np.sqrt(weights)` for fweights,
`leverage / weights` in HC2/HC3) and `_X_untransformed`.

Decide and document ONE scale for stored results: recommended
`residuals` = response-scale (unweighted) residuals + `weights` stored
separately; anything that needs sqrt-weight-scale residuals (several meat
matrices) computes them locally. `_u_hat` needs no compat alias (see
resolved decision 5). The public contract is already response-scale:
`Feols.resid()` divides by `sqrt(weights)` today
(`_result_accessor_mixin.py`), so after 3b it returns the stored residuals
directly and that division is deleted. Pin `resid()` outputs in the golden
tests before starting 3b anyway.

Risk: every intermediate array changes ⇒ gate on the golden tests +
`test_vs_fixest` (coefficients/SEs must match to machine precision, they
are algebraically identical). Do 3b per-estimator (OLS → IV → IWLS), not
in one PR.



Effort: large but incremental — one vcov estimator / one method per PR. Risk: low per PR, guarded by golden tests.



---



## Step 4 — Slim the model classes



With step 3 done, `Feols` is a results container + orchestrator. Now:



1. Move post-estimation method bodies (`wildboottest`, `ccv`, `decompose`,

`ritest`, `fixef`, `predict`, `wald_test`) into `post_estimation/` modules

(some already live there); keep thin bound methods for the public API.

2. Keep `_result_accessor_mixin` as the single home of `tidy/coef/se/...` —

delete the duplicated implementations in `FixestMulti` (have it build a tidy

DataFrame from its models via one helper instead of 6 methods).

3. Move the huge docstring *examples* from `api/feols.py` etc. into the docs

(quartodoc pages), keeping concise parameter docs in code. `api/feols.py`

drops from ~600 to ~150 lines.

4. Target sizes: `feols_.py` < 700 lines, `fepois_.py` < 200 (family definition),

`feglm_.py` < 400.



Effort: medium, mechanical. Risk: low.



---



## Step 5 — Refactor `FixestMulti` into plan + runner + results



Replace the string-typed if-chains and `model_map[(method, is_iv)]` with explicit

structure:



```

@dataclass(frozen=True)

class ModelSpec: # everything needed to fit ONE model

formula: Formula

model_cls: type[Feols]

sample_split: ...

estimation_options: ... # demeaner, solver, collin_tol, family kwargs



def expand_specs(fml, data, split, fsplit, method, ...) -> list[ModelSpec] # pure

def fit_one(spec, data, cache) -> Feols # uses steps 2-4

class FixestMulti: # thin results container

models: dict[str, Feols]

tidy/etable/coefplot/vcov... (delegating)

```



1. `expand_specs` absorbs `_prepare_estimation` + the triple loop's bookkeeping —

pure and unit-testable without fitting anything.

*(Update: implemented in `estimation/plan_.py` — `ModelSpec`, `MODEL_REGISTRY`,
`expand_specs` (absorbs the triple loop incl. split expansion and demean-cache
block boundaries via `ModelSpec.cache_key`), and `fit_one` (single pipeline).
`_prepare_estimation` deliberately kept as the thin "parse formula + store
options" step since all four api modules call it; folding it into
`expand_specs` is the remaining item 4 below, best done together with the
api-layer rewrite.)*

2. A **model registry** (`method -> (model class, extra-kwargs builder)`) replaces

the five `if self._method in {...}` blocks. Quantreg/QuantregMulti register like

everyone else.

3. `fit_one` owns the fit pipeline (`prepare_model_matrix → get_fit → vcov →

inference → performance → clear`), currently inlined in the loop. One place,

one order, for all model types.

4. `api/feols.py` becomes: validate → expand_specs → fit each → wrap. The

single-model-vs-multi return logic stays identical.



Effort: medium-large. Risk: medium — this is the most structural change; do it after steps 2-4 so the pieces being orchestrated are already clean.



---



## Step 6 — Numpy-style API for Feols / Fepois / Feglm



Now nearly free, because step 3 created array-based cores and step 5 separated

formula handling from fitting:



```

fit = Feols.from_arrays(y, X, fe=None, weights=None, coefnames=None,

vcov="iid", demeaner=None, ssc=None, ...)

```



1. Implement as a classmethod that builds the minimal `FitState` directly from

arrays and runs the same `fit_one` pipeline minus the formulaic step.

Formula path = `model_matrix(...)` + `from_arrays(...)` — the formula API

becomes a thin layer over the numpy API, guaranteeing the two never diverge.

2. Post-estimation that needs the original DataFrame (`ritest`, `fixef()`,

`predict(newdata=...)`) raises a clear "not available without formula/data"

error — same pattern as `lean=True` today.

3. Add `Fepois.from_arrays` / `Feglm.from_arrays` via the shared IWLS core.

4. Document as the supported way to plug pyfixest into sklearn-style pipelines

and simulations.



Effort: small-medium once 3+5 are done. Risk: low (purely additive).



---



## Step 7 — Layout, naming, polish



Final pass once the dust settles:



1. Module naming: drop the trailing-underscore convention

(`FixestMulti_.py` → `multi.py`, `models/feols_.py` → `models/ols.py`, ...),

keeping import shims for one release if anything external imports them.

Target tree:



```

pyfixest/

estimation/

api/ # feols.py, fepois.py, feglm.py, quantreg.py (thin)

formula/ # parse.py, model_matrix.py, factor_interaction.py

core/ # pure numpy: ols.py, iwls.py, vcov.py, inference.py, solvers.py

demean/ # demeaners.py (configs+strategies), cache.py, torch/

models/ # ols.py, iv.py, glm.py (incl. poisson), quantreg.py

multi.py # ModelSpec, expand_specs, registry, FixestMulti

post_estimation/

core/ # rust bindings only

```



2. Enforce import direction (api → multi → models → core/demean → rust) with an

import-linter contract in CI, so layering can't silently regress.

3. Tighten typing (mypy/pyright on `estimation/`), remove `# type: ignore`s that

the new structure makes unnecessary.

4. Prune `pyfixest/__init__.py` lazy-import table to the intended public surface.



---



## Sequencing and release mapping (suggestion)



| Release | Steps | Notes |

|---|---|---|

| 0.61 | 0 + 1 | pure deletion; announce removals in changelog |

| 0.62 | 2 | demeaner stack collapse |

| 0.63–0.64 | 3 + 4 | core extraction, model slimming (many small PRs) |

| 0.65 | 5 | FixestMulti rework |

| 0.66 | 6 | numpy API (new feature) |

| 0.67 | 7 | naming/layout, numba backend removal if deprecated in 0.61 |



Guiding rules for every PR: no user-facing API change, golden tests + `test_vs_fixest` green, each PR reviewable in < ~500 changed lines where possible.



## Open decisions



1. Numba MAP backend: deprecate now (recommended) or keep as optional extra?

2. `model_matrix_fixest` public export: remove immediately or warn one release?

3. Removed kwargs (`use_compression`, `demeaner_backend`): hard removal vs. one release of raising a helpful `TypeError`-style message?

4. Should `Fepois` be folded into `Feglm` (family="poisson") internally (recommended in step 3), keeping the `Fepois` class name as a thin subclass for back-compat? *(Update: step 3 extracted both loops as-is into `fit_iwls_poisson` / `fit_iwls_glm`; the fold is now a contained, separately testable diff.)*

5. ~~Are `fit._X`, `fit._Y`, `fit._u_hat` de-facto public?~~ **Resolved:** de-facto public but not supported — users are directed to `fit.resid()` and friends. No backwards compatibility for underscore attributes; rename freely in 3a/3b. Public method outputs (`resid()`, `coef()`, `predict()`, `tidy()`) are the contract and must be pinned by golden tests.
