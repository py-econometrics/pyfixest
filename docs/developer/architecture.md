# Pyfixest architecture

This document is the authoritative overview for contributors and coding agents.
Preserve public behavior and numerical correctness while extending pyfixest.

## Contents

- [Design principles](#design-principles)
- [Stable core](#stable-core): the contracts that need design approval
- [Estimator add-ons](#estimator-add-ons): where a new estimator lives
- [Estimation flow](#estimation-flow): the shared pipeline diagram
- [Formula-state and lifecycle boundaries](#formula-state-and-lifecycle-boundaries)
- [Estimation-state vocabulary](#estimation-state-vocabulary): the terms new
  shared-core work must use
- [Implemented array and weight domains](#implemented-array-and-weight-domains):
  the frozen state values and their contracts
- [Repository map and extension seams](#repository-map-and-extension-seams)
- [Result and numerical boundaries](#result-and-numerical-boundaries): the
  support-matrix requirement
- [Public documentation](#public-documentation)
- [Compatibility changes](#compatibility-changes)

## Design principles

1. Mirror R `fixest` behavior, names, and defaults unless an intentional
   difference is documented and tested.
2. Mirror the nearest existing pyfixest implementation before inventing a new
   pattern.
3. Keep the shared estimation core narrow. New estimators are add-ons composed
   from stable primitives.
4. Model methods orchestrate. Standalone functions perform numerical work.
5. Put only measured, non-vectorizable hot loops in Rust.

## Stable core

The stable core contains:

- fixest-style formula parsing and model-matrix construction;
- estimation configuration and multiple-estimation planning;
- demeaning, collinearity handling, weight transforms, and solver primitives;
- generic vcov and inference primitives;
- fitted-result interfaces and multiple-estimation containers;
- backend contracts and native kernels.

Obtain maintainer design approval before changing these contracts, including
expanding the core for one estimator. Implementation changes that preserve the
contracts follow normal implementation and verification; they do not need a
separate design approval.

## Estimator add-ons

A new estimator starts as a standalone public function in
`pyfixest/estimation/api/` or the relevant domain package, such as
`pyfixest/did/`. Its estimator-specific preparation, iteration, and
post-processing stay in its own modules. It may call stable formula, data,
demeaning, solver, vcov, and reporting primitives, but must not add
estimator-specific branches to `runner.py`, `plan_.py`, or generic model
classes merely for convenience.

Reuse an existing result class only when its semantics and supported operations
are genuinely compatible. Otherwise create a dedicated result class with the
same common accessors where applicable.

A helper moves into shared internals only when it has a current shared consumer
and a generic contract. Hypothetical future reuse is not sufficient.

## Estimation flow

Estimators using the shared estimation pipeline build an `EstimationConfig`.
`parse_formula` expands multiple-estimation syntax, and
`runner.run_estimation` and `fit_one` prepare each model before dispatching to
its estimator-specific fit path. DiD estimators use their own domain-specific
entry points and orchestration.

```text
shared estimator API
  -> EstimationConfig
  -> parse_formula
  -> prepare_model_matrix
  -> estimator-specific get_fit
       -> feols / feiv
            -> construct within-scale arrays
            -> drop collinear columns
            -> solve OLS / 2SLS with local weight transforms
       -> fepois / feglm
            -> IRLS with explicit observation and working weights
       -> quantreg
            -> Frisch-Newton solve (absorbed fixed effects are unsupported)
  -> vcov
  -> get_inference
  -> fitted result / FixestMulti
```

`FixestMulti` is a container for fitted results. Numerical behavior belongs in
the individual models and shared primitives, not in the container.

## Formula-state and lifecycle boundaries

`ModelMatrix` builds the formula inputs and is also the formula state a fitted
model retains. Missing, infinite, singleton, and other formula-level row filters
run during construction; afterwards the instance is treated as read-only, and
its dependent, independent, fixed-effect, IV, weight, and offset roles stay on
formula scale. Later transformations produce separate within- or solver-scale
arrays without changing the retained formula state. Estimator-level filters
that need the materialized design, such as GLM separation, call
`ModelMatrix.without_rows()`, which returns a filtered copy whose
`dropped_row_index` includes the dropped rows, so the canonical row sample and
the demeaning-cache key stay aligned with the data that enter IRLS.
`store_data=False` and `lean=True` discard the formula state together with the
other retained input state.

The generic runner operates on the structural `FittedModel` protocol. Response
validation, estimator-specific post-fit work, and result expansion live behind
model hooks rather than estimator-name dispatch in the runner. Pipeline-object
constructors only assemble configuration and child objects; for example,
`QuantregMulti` prepares its children in `prepare_model_matrix`, not during
construction.

`DemeanCache` shares named, read-only array entries within a multiple-estimation
cache block. Its key is the dropped-row index set; the block shares fixed
effects and observation weights. GLM iterations share a preconditioner cache,
but do not cache demeaned values because their working weights change.

After fitting, inference, and estimator-specific completion, `_clear_attributes()`
applies the retention policy. Post-estimation paths use `require_retained` to
fail informatively when required state was removed. The public `vcov()` method
remains an explicitly in-place post-fit operation.

## Estimation-state vocabulary

Name the transformation domain explicitly in shared-core work:

| Term | Meaning |
|---|---|
| formula data | Post-filtering, formula-materialized tabular values and metadata; not necessarily identical to raw user columns |
| observation weights | The weights supplied by the user, together with their `aweights` or `fweights` interpretation |
| within data | Arrays after the possibly weighted FE projection, still in original units and not premultiplied by square-root weights |
| solver data | Ephemeral arrays such as `design_sqrt_weighted` used by a numerical solve |
| working state | GLM iteration values, including a working response and working weights, kept distinct from observation weights |
| response residual | A residual in the response's units; weighted scores and solver residuals should be named separately |

These states should be completed values returned by transformations. Persisted
arrays should not change type or numerical domain, and solver scratch should
not replace canonical within data. A fitted result may still expose explicitly
in-place post-estimation operations, but those operations must not repurpose
estimation fields.

## Implemented array and weight domains

The shared linear and GLM paths use frozen, slotted state values. Frozen state
prevents field rebinding, but contained NumPy arrays remain mutable unless
they are explicitly marked read-only:

| State | Persisted contract |
|---|---|
| `ModelMatrix` | Formula-materialized pandas tables remain on formula scale and keep dependent, independent, fixed-effect, IV, weight, and offset roles separate. |
| `ObservationWeights` | Canonical user-scale weights and their `aweights` or `fweights` semantics. `values=None` is the allocation-free unweighted path. |
| `WithinLinearData` | Unpremultiplied within-scale response and design arrays. |
| `WithinIvData` | Extends `WithinLinearData` with the instrument and endogenous arrays that only IV models carry. |
| `GlmWorkingState` | Final within-scale working response and design, IRLS working weights, predictors, means, and response- and working-residual domains. |
| `SandwichComponents` | Weighted scores, Hessian, and bread built by the OLS, 2SLS, and IRLS fit primitives for covariance calculations. The 2SLS fit scores the first-stage projection of the design, so every estimator shares one sandwich form. |
| `DemeanedData` | Array-native cache entries whose ordered column names are metadata rather than DataFrame conversions around each reuse. |

Analytic weights keep the retained row count as the effective sample size;
frequency weights use the sum of their user-scale values. Probability weights
(`pweights`) remain unsupported. Fixed-effect projection may use observation or
IRLS weights, but the returned arrays remain within scale. OLS, IV, and IRLS
fit primitives create square-root-weighted design and response arrays only as
local solver temporaries. They persist response-unit residuals and weighted
scores or cross-products, not solver-scale copies of canonical data.
Singleton fixed-effect detection counts physical rows even under frequency
weights, as in fixest: an aggregate row alone in its level is dropped although
its literal expansion would not be.

GLMs keep two weight concepts deliberately separate. `ObservationWeights`
never changes after formula preparation, while each IRLS iteration computes
working weights and the final values live in `GlmWorkingState`. Response
residuals and working residuals likewise have separate fields.

A post-estimation path states which estimators, weighting schemes, and design
features it can represent, and rejects the rest. Declare support in the result's
`Capabilities`, check it before reading estimation state, and raise
`NotImplementedError` naming the unsupported combination.
Reinterpreting one estimator's arrays as another estimator's domain, such as
reading GLM working state or a quantile solver's output as linear-model arrays,
is a silently wrong result rather than a fallback. A path whose refits cannot
yet replay the original estimation contract rejects the estimator until they
can.

An operation that cannot reconstruct the complete state of a fitted result
returns its value instead of mutating the result in place.

## Repository map and extension seams

Paths below are relative to `pyfixest/` unless shown otherwise.

| Change | Primary location | Pattern to follow |
|---|---|---|
| Estimator API | `estimation/api/` or the domain package | nearest API/result pair; tests, exports, and quartodoc registration |
| Model/result type | `estimation/models/<name>_.py` | nearest compatible result class |
| Post-estimation | `estimation/post_estimation/` | `ritest.py` plus the thin `Feols.ritest` wrapper |
| Shared numerical primitive | `estimation/internals/` | `fit_.py`, `vcov_.py`, or nearest analogue |
| Formula behavior | `estimation/formula/` | existing parser/model-matrix seams |
| Configuration and orchestration | `estimation/config.py`, `plan_.py`, `runner.py` | typed options and generic model hooks |
| Demeaner configuration | `demeaners.py` | existing public configurations |
| Rust kernel | `core/` and repository-root `src/` | `src/nw.rs` → `core/nw.py` |
| DiD, reporting, utilities | `did/`, `report/`, `utils/` | nearest domain implementation |
| Tests and user documentation | repository-root `tests/` and `docs/` | nearest test matrix and tutorial/how-to |

Public estimation functions use one module per entry point. Model modules end
in `_` so they do not shadow public functions. Compatibility shims in the
`estimation/` root are not implementation locations.

- **Vcov type:** literal in `internals/literals.py`, parsing in
  `VcovSpec.from_user_input` (run before fitting), model support in
  `_check_vcov_support`, a small dispatch method in `_vcov_from_spec`, math in
  `internals/vcov_utils.py`, `internals/vcov_.py`, or Rust, and wiring through
  `FixestMulti`/quantreg where supported. Follow NW/DK HAC.
- **Estimation-time option:** shared typed alias in `internals/literals.py`, API
  validation, `EstimationConfig`, and `plan_._build_model_kwargs`.
- **Rust kernel:** implementation in `src/<topic>.rs`, registration in
  `src/lib.rs`, stub in `core/_core_impl.pyi`, and wrapper in `core/`. Keep a
  readable NumPy reference where feasible. Reserve Rust for measured,
  non-vectorizable hot loops; use ordinary NumPy elsewhere.

Reuse formula handling, `capture_context`, `_narwhals_to_pandas`, cluster
preparation, `run_crv_loop`, and `_create_rng` rather than rederiving them.

## Result and numerical boundaries

A model method validates inputs, unpacks model state, calls a module-level
function with keyword arguments, and stores or returns the result. Numerical
functions operate on arrays and return small typed dataclasses whose docstrings
state array shapes. Keep functions single-purpose; splitting a solver loop
should not obscure the algorithm or hurt compilation.

Every estimator or inference feature specifies and tests behavior for
`aweights`, `fweights`, fixed effects, IV, multiple estimation, `lean=True`,
`store_data=False`, and relevant backends. Unsupported combinations fail
explicitly. Post-estimation code must reject stripped-data paths it cannot
support; silent fallback is never acceptable.

## Public documentation

Public functions, methods, and classes need NumPy docstrings with complete
Parameters/Returns, an executable `{python}` example, root-relative `.qmd`
links, and a linked paper for econometric methods. New public functions/classes
need exports and quartodoc registration. User workflows usually need a
`docs/how-to/` guide or an extension to the nearest guide; documentation ships
with the feature.

Add a one- or two-line `docs/changelog.qmd` entry for features, behavior/default
changes, bug fixes, deprecations, performance changes, or new contributor
tooling. Internal refactors, guidance edits, CI tweaks, and typo fixes need no
entry. Never hand-edit generated `docs/reference/**`.

## Compatibility changes

Fixest parity is the default. Record intentional differences in
[fixest-compatibility.md](fixest-compatibility.md) with their rationale and
tests. When compatibility cannot be preserved directly, use a reviewed
deprecation path.
