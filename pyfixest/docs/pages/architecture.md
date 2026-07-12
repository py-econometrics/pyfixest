<!-- Generated from ARCHITECTURE.md; do not edit. -->

# PyFixest architecture

This is the canonical, repository-local guide to how PyFixest is assembled.
It describes the current code, not a desired future design. Update it in the
same pull request whenever an architectural boundary, estimation stage, model
capability, cache lifetime, or Python/Rust interface changes.

PyFixest mirrors R's `fixest` at the user boundary. Internally it separates
public argument handling, immutable estimation intent, formula expansion,
execution, model-specific numerics, inference, and reporting. Most regressions
flow through the shared planner and runner; difference-in-differences is a
separate composition layer built on fitted regression objects.

## Dependency direction

Read the arrows as “may call or depend on.” Lower layers must not depend on
higher layers merely for convenience.

```text
pyfixest/__init__.py (lazy public facade)
        |
        +--> estimation/api/ -----------+
        |                               |
        |                    config.py + plan_.py + runner.py
        |                               |
        |                         models/ + quantreg/
        |                           /          \
        |              formula/ + internals/  post_estimation/
        |                           |
        |                    demeaners.py + core/
        |                                      |
        |                               core._core_impl
        |                                      |
        |                                    src/ (Rust)
        |
        +--> did/ ------> public regression API and fitted results
        +--> report/ ---> fitted results (model wrappers import report lazily)
        +--> utils/ ----> shared conversion, context, data, and DGP helpers
```

The important ownership rules are:

- `estimation/api/` owns public signatures, argument validation, defaults, and
  creation of `EstimationConfig`. It should not contain estimator numerics.
- `config.py` is the immutable record of one public estimation request.
- `plan_.py` parses/expands that request into model specifications and maps
  method identifiers to result classes. It does not own the runtime loop.
- `runner.py` prepares data, owns the model loop and cross-model cache lifetime,
  and unwraps a one-model result.
- `models/` owns model state and short pipeline methods. Shared numerical work
  belongs in `internals/`; substantial post-estimation work belongs in
  `post_estimation/`.
- `formula/` is the only place that should extend fixest formula parsing or
  Formulaic model-matrix behavior.
- `core/` is the typed Python boundary around compiled kernels in `src/`.
- `FixestMulti` is a results container, not an estimator superclass or an
  execution engine.
- `report/` consumes fitted results. Result methods that expose reporting must
  remain thin/lazy so importing estimation does not create an eager cycle.

`Feiv.first_stage()` is an intentional callback into the public `feols` path:
the first stage is itself an OLS fit and should receive the same public
validation and fitting behavior. Do not introduce additional upward
dependencies without documenting why a lower-level helper cannot be used.

## Repository ownership map

| Area | Responsibility | Typical reason to change it |
|---|---|---|
| `pyfixest/estimation/api/` | Public estimation functions and early checks | New entry point or user option |
| `pyfixest/estimation/config.py` | Immutable estimation request | Thread a new estimation-time option |
| `pyfixest/estimation/plan_.py` | Formula parse result, model registry, spec expansion, one-model orchestration | Add a model family or constructor option group |
| `pyfixest/estimation/runner.py` | Data/context/split preparation, cache blocks, result assembly | Change execution order or cross-model reuse |
| `pyfixest/estimation/formula/` | Fixest syntax and Formulaic model matrices | Add formula syntax or design-matrix state |
| `pyfixest/estimation/models/` | Fitted-model state, model-specific fit and inference dispatch, thin methods | Add model behavior or a thin post-estimation wrapper |
| `pyfixest/estimation/internals/` | Solvers, fit records, covariance math, demeaning cache, collinearity, separation | Shared numerical or estimation logic |
| `pyfixest/estimation/post_estimation/` | Substantial operations on fitted models | New post-estimation procedure |
| `pyfixest/estimation/quantreg/` | Quantile solvers, multi-quantile process, quantile covariance | Quantile-specific algorithm or inference |
| `pyfixest/demeaners.py` | Typed strategy objects for MAP and LSMR backends | Add or configure a demeaning backend |
| `pyfixest/core/` + `src/` | Python wrappers/type stubs + PyO3/Rust hot loops | Performance-critical kernel |
| `pyfixest/did/` | TWFE, DID2S, saturated event study, and local projections | Difference-in-differences behavior |
| `pyfixest/report/` | Tables, summaries, coefficient/event-study plots | Output formatting or visualization |
| `tests/` | Public integration, external references, kernel parity, edge cases | Verify all of the above |

## Estimation flow

### 1. Public API records intent

`feols`, `feglm`, and `quantreg` validate their user-facing inputs, resolve
applicable defaults such as SSC and the demeaner, and construct a frozen
`EstimationConfig`. `fepois` is a convenience facade over
`feglm(..., family="poisson")`. Each API then calls `parse_formula(config)` and
`run_estimation(config, parsed)`.

The configuration is deliberately flat: it carries shared fields plus GLM-only
and quantile-only fields. A model class should not reach back into an API
function to recover an option.

### 2. Formula parsing and planning expand the request

`plan_.parse_formula` delegates fixest syntax to
`formula.parse.Formula.parse_to_dict` and returns `ParsedFormula`:

- `formula_dict` groups expanded formulas by fixed-effects specification;
- `is_iv` records whether any formula contains a first stage; and
- `is_multiple_estimation` includes formula expansion, sample splitting, and a
  list of quantiles.

`runner._split_plan` determines full-sample and split runs. `build_all_splits`
orders the full sample first for `fsplit`, followed by sorted split values.
`expand_specs` then walks split value, fixed-effect key, and formula in that
order. It creates one immutable `ModelSpec` per fit and resolves its class from
`MODEL_REGISTRY`. `_build_model_kwargs` threads only the constructor option
groups declared in the registry entry's `needs` set.

This ordering is part of the cache contract: specifications with the same
`(sample_split_value, fixef_key)` stay contiguous.

### 3. The runner executes cache blocks

`runner.run_estimation` converts supported dataframe inputs to pandas through
Narwhals, optionally copies them, resets the index, captures Formulaic context,
and constructs an initially empty `FixestMulti` container.

The runner owns two dictionaries for the current cache block: demeaned columns
and LSMR preconditioners. When a `ModelSpec.cache_key` changes, both dictionaries
are replaced. For each spec, `fit_one`:

1. injects the current cache dictionaries into constructor arguments;
2. constructs the resolved model class;
3. calls `prepare_model_matrix()`;
4. validates a GLM dependent variable when applicable;
5. calls the model's `get_fit()`;
6. computes covariance and inference unless the design has no covariates;
7. computes OLS performance/Wald results or IV first-stage results as applicable;
8. applies `store_data`/`lean` cleanup; and
9. returns the fitted model for insertion into `FixestMulti.all_fitted_models`.

`QuantregMulti` is expanded into its individual quantile models before
insertion. If the parsed request represents multiple estimation, the populated
container is returned. Otherwise `fetch_model(0, print_fml=False)` unwraps the
single model.

### 4. Model classes own method-specific fitting

`Feols.prepare_model_matrix()` creates the Formulaic-backed `ModelMatrix`,
aligns rows after missing-value/singleton handling, records coefficient and
fixed-effect metadata, stores the reusable Formulaic `ModelSpec`, and initializes
weights and observation counts.

The model-specific fit branches are:

- **OLS/WLS (`Feols`)**: demean `Y`/`X` when fixed effects are present, convert
  to arrays, remove collinear columns, apply the square-root weight transform,
  and call `internals.fit_.fit_ols`.
- **IV/2SLS (`Feiv`)**: extend the same path to endogenous variables and
  instruments, then call `internals.fit_.fit_iv`. After second-stage inference,
  fit the first stage through `feols` and compute weak-instrument diagnostics.
- **GLM/Poisson (`Feglm` and subclasses)**: apply model-specific dependent-
  variable and separation checks, convert arrays, and call
  `internals.fit_glm_.fit_glm_irls`. Fixed-effect residualization is supplied as
  an IRLS callback, so working variables can be demeaned each iteration.
- **Quantile regression (`Quantreg`, `QuantregMulti`)**: prepare a design
  without fixed effects, run the Frisch-Newton/preprocessed solver, and compute
  quantile-specific covariance. Multi-quantile algorithms share starting
  information across quantiles before the runner flattens their results.

Every branch writes the common result vocabulary expected by inference and
reporting: coefficient names and estimates, residuals/scores, a Hessian or
cross-product matrices, fitted values, and method metadata. Shared covariance
dispatch then builds the bread/meat, applies SSC, and calls `get_inference()`.

## Model hierarchy and capability matrix

Inheritance provides shared result and covariance machinery; it does not mean
every inherited method is supported. Capability flags and public validation
must reject invalid combinations rather than silently produce numbers.

| Public call / result | Fixed effects | IV | User weights | Important capability boundary |
|---|---:|---:|---:|---|
| `feols` → `Feols` | Yes | No | Analytic/frequency | Main OLS/WLS path; HC2/HC3 require no fixed effects; supports CRV and HAC families |
| IV `feols` → `Feiv(Feols)` | Yes | Yes | Analytic/frequency | 2SLS plus first stage; HC2/HC3 and CRV3 are rejected |
| `fepois` → `Fepois(Feglm)` | Yes | No | Analytic/frequency | Poisson IRLS, separation checks, optional offset; no IV |
| `feglm` → `Fegaussian`, `Felogit`, or `Feprobit` | Yes | No | Analytic/frequency | Family-specific IRLS; no IV; decomposition/CCV are disabled |
| `quantreg` → `Quantreg(Feols)` | No | No | No | `iid`, `hetero`/HC aliases, `nid`, or one-way CRV1; CRV3/HAC disabled |
| Multiple expansion → `FixestMulti` | Per contained model | Per contained model | Per contained model | Pure ordered result container with aggregate access/report methods |

`event_study`, `did2s`, and `lpdid` live outside this registry. They compose
regression fits and overwrite or aggregate inference appropriate to the DID
estimator. `event_study` dispatches among TWFE, DID2S, and saturated designs;
`lpdid` returns its dedicated local-projections result.

When widening a capability, define behavior for fixed effects, analytic and
frequency weights, IV, multiple estimation, `store_data=False`, and `lean=True`.
An inherited method is not evidence that all of those paths are valid.

## Data, state, and cache lifecycle

### Input data

The runner converts pandas, Polars, and DuckDB-compatible inputs to pandas and
resets to a clean `RangeIndex` because Formulaic and subsequent row drops must
remain aligned. With `copy_data=True` (the default), this work is isolated from
the caller. `copy_data=False` permits index and formula-generated-column side
effects and is the only supported exception to the no-mutation rule.

Each fitted model begins with a working dataframe. Model-matrix preparation
materializes pandas `Y`, `X`, optional `Z`, fixed effects, weights/offset, the
rows removed for missingness, and a Formulaic model specification. Fitting turns
the numerical pieces into NumPy arrays and adds estimates, residuals, fitted
values, scores/cross-products, covariance, degrees of freedom, and inference.

After inference, `_clear_attributes()` applies memory policy:

- `store_data=False` removes the retained estimation dataframe. Data-dependent
  covariance updates and other post-estimation operations must therefore be
  requested during fitting or fail with actionable guidance.
- `lean=True` additionally removes working design arrays, demeaned arrays,
  cluster data, scores, residuals, fitted values, weights, and IV cross-products.
  Already-computed compact coefficient/inference results remain, but methods
  must not assume the deleted working state exists.

Do not make new post-estimation code depend on an incidental private attribute.
List the state it requires, validate that state at the public method boundary,
and define behavior for both memory flags.

### Cross-model demeaning caches

`runner.run_estimation`, not `FixestMulti`, owns cache lifetime. The outer cache
key is `(sample_split_value, fixef_key)`. Inside one block, every model receives
the same dictionaries through its `DemeanCache`:

- `lookup_demeaned_data` maps a frozen missing-row index to a dataframe of
  already-demeaned columns. A later specification demeans only columns not yet
  present for that row sample.
- `lookup_preconditioner` maps the same row index to the first reusable LSMR
  preconditioner. IRLS calls reuse it across iterations; IV first-stage fitting
  explicitly passes a compatible cached preconditioner into its OLS fit.

The dictionaries reset when the sample split or fixed-effects formula changes.
They are never global, never shared across unrelated public calls, and must not
be reused for an incompatible row/fixed-effect design.

## Python/Rust boundary

Rust is reserved for measured hot loops that plain NumPy cannot express
efficiently. Current kernels cover collinearity detection, CRV1 accumulation,
MAP/within demeaning, singleton detection, nested fixed-effect counts, and
Newey-West/Driscoll-Kraay meat matrices.

The interface has four explicit layers:

1. `src/<topic>.rs` owns the PyO3 function or class and validates assumptions
   that must be safe inside Rust.
2. `src/lib.rs` registers it in the `_core_impl` Python extension module.
3. `pyfixest/core/_core_impl.pyi` declares the Python-visible NumPy dtypes,
   shapes/return structure, and class properties for static tooling.
4. `pyfixest/core/<topic>.py` exposes a clean alias or a small validation/dtype
   normalization wrapper. Higher estimation layers import this Python surface,
   not raw private Rust names.

For a new kernel, mirror an end-to-end precedent such as
`src/nw.rs → src/lib.rs → core/_core_impl.pyi → core/nw.py`. Keep a readable
NumPy or brute-force reference in tests. Compare results over normal, empty or
minimal, singleton/collinear, reordered, and dtype-boundary cases before adding
a benchmark. If the extension fails to import after a Rust change, run the
documented maturin import-hook setup through `pixi`; do not bypass the project
environment with a bare Python build.

## Numerical and structural test map

| Contract | Primary tests | Strong reference |
|---|---|---|
| Public signatures, data conversion, context, memory flags | `tests/test_api.py`, `tests/test_errors.py` | Public calls and explicit failures |
| Formula parsing and multiple-estimation planning | `tests/test_formula_parse.py`, `tests/test_formulas.py`, `tests/test_plan.py` | Parsed known cases and legacy/public parity |
| OLS/WLS estimates and inference | `tests/test_vs_fixest.py`, `tests/test_wls_types.py`, `tests/test_ses.py` | R `fixest` matrix over formulas/vcov/weights/SSC |
| IV fitting and diagnostics | `tests/test_iv.py` | R/reference statistics and seeded designs |
| GLM and Poisson | `tests/test_feols_feglm_internally.py`, `tests/test_poisson.py` | Dummy-expanded OLS/GLM identities and R `fixest` |
| Quantile regression | `tests/test_quantreg.py` | R `quantreg`, Statsmodels, and stored values |
| Demeaners and cache/preconditioner reuse | `tests/test_demean.py`, `tests/test_api.py` | `pyhdfe`, backend parity, reuse invariants |
| CRV and HAC kernels | `tests/test_crv1.py`, `tests/test_crv1_vcov.py`, `tests/test_hac_meat.py`, `tests/test_hac_vs_fixest.py` | NumPy/brute force plus R `fixest`/`sandwich` |
| Collinearity, singleton, nested FE kernels | `tests/test_collinearity.py`, `tests/test_detect_singletons.py`, `tests/test_count_fixef_fully_nested.py` | Python reference and known cases |
| Prediction/fixed effects and post-estimation | `tests/test_predict_resid_fixef.py`, `tests/test_ritest.py`, `tests/test_decomposition.py`, `tests/test_multcomp.py` | Public integration, stored/R/brute-force references |
| Reporting and plots | `tests/test_summarise.py`, `tests/test_plots.py` | Stable dataframe/text fields and plot structure |
| DID estimators | `tests/test_did.py`, `tests/test_event_study.py` | R packages, packaged fixtures, and estimator invariants |

Use the narrowest relevant test during development, then the quick suite. R and
HAC comparisons use strict markers and dedicated pixi tasks; new files that
import `rpy2` must be registered in `tests/conftest.py`. Numerical work requires
a known-value, brute-force, external-package, or simulation reference—not only
shape checks.

## Routing common changes

- **Post-estimation feature:** computation in `post_estimation/`; thin model
  method; use `ritest.py` and `Feols.ritest` as the pattern.
- **Covariance estimator:** literal/input parsing, small dispatch in
  `Feols.vcov`, shared bread/meat in `internals/vcov_.py` or `vcov_utils.py`,
  SSC and multi-model threading, then external/brute-force reference tests.
- **Estimation-time option:** public signatures/checks, `EstimationConfig`, and
  `_build_model_kwargs`; use `MODEL_REGISTRY.needs` for family-specific options.
- **Public function:** one module under `estimation/api/`, export through the API
  package and lazy top-level facade, register with quartodoc, and test the public
  import end to end.
- **Formula syntax:** parser/model-matrix code only; test parsing and a public fit.
- **Rust kernel:** all four boundary layers above plus a Python reference test.

For exact coding, documentation, and commit workflow rules, continue with the
repository-local `AGENTS.md` and `.agents/feature-pr.md` files.
