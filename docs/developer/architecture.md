# Pyfixest architecture

This document is the authoritative overview for contributors and coding agents.
Preserve public behavior and numerical correctness while extending pyfixest.

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

Changes to these contracts affect multiple estimators and require maintainer
design approval before implementation.

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

## Historical estimator-state lifecycle (pre-refactor)

Before the immutable-state refactor, fitted-model classes used themselves as
both a work area and a result object. Several private attributes therefore
changed representation and numerical scale while a model was fitted. This
section preserves that baseline as migration history; it is not the current
contract for new code.

For linear models, the transformations are:

```text
formula-materialized scale
  -> weighted fixed-effect projection
within scale (original units; not premultiplied)
  -> multiplication by sqrt(observation weights)
solver scale
```

The word *weighted* has two distinct meanings in that sequence. Fixed-effect
residualization uses the observation weights when computing group projections,
but its output is still in the dependent variable's and covariates' original
units. Only `wls_transform()` premultiplies those within-scale values by the
square root of the weights.

### Linear and IV models

`Feols.prepare_model_matrix()` initially stores formula-materialized pandas
objects in `_Y`, `_X`, `_fe`, `_weights_df`, and, for IV models, `_Z` and
`_endogvar`. It also retains a copy of the response in `_Y_untransformed`.
The subsequent stages reuse the same model object:

| Stage | OLS fields | Additional IV fields | Representation and scale |
|---|---|---|---|
| `demean()` | `_Yd`, `_Xd` | `_Zd`, `_endogvard` | pandas DataFrames on within scale; the FE projection uses observation weights |
| `to_array()` | `_Y`, `_X` are reassigned | `_Z` and `_endogvar` are reassigned | NumPy arrays; `_Y` and `_X` now hold within-scale values |
| `drop_multicol_vars()` | `_X` may lose columns | `_Z` may lose columns | NumPy arrays on within scale; coefficient-name lists mutate in parallel |
| `wls_transform()` | `_Y`, `_X` are reassigned again | `_Z` and `_endogvar` are reassigned again | NumPy arrays on solver scale, premultiplied by `sqrt(_weights)` |
| solve | `_Z` aliases `_X`; fit products overwrite empty placeholders | IV cross-products and fit outputs replace placeholders | Solver-scale arrays remain attached to the fitted result |

Consequently, the meaning of `_X`, `_Y`, and `_Z` cannot be inferred from
their names or type annotations alone. OLS and IV `_u_hat` are also stored on
solver scale; the public `resid()` accessor divides by `sqrt(_weights)` to
return residuals in response units. Consumers such as leverage, covariance,
fixed-effect recovery, and decomposition either expect solver-scale fields or
undo the transform locally.

The API currently accepts analytic weights (`aweights`) and frequency weights
(`fweights`), not probability weights. Both weight types use the same weighted
point-estimation transform. They differ in effective sample size and parts of
inference: analytic weights use the number of retained rows, while frequency
weights use the sum of the weights. Estimator-specific support limitations
must remain explicit rather than being inferred from the array representation.

### GLM, Poisson, and quantile models

`Feglm` starts with formula-materialized DataFrames and converts `_Y`, `_X`,
and `_fe` to arrays before IRLS. Each iteration creates a working response and
working weights, performs a weighted FE projection, and solves a square-root-
weighted least-squares problem. After the final iteration:

- `_X` and `_Y` are replaced by the final solver-scale working design and
  response;
- `_weights`, which initially contains observation weights, is replaced by the
  final IRLS weights and duplicated in `_irls_weights`;
- `_u_hat` is the solver-scale working residual, while
  `_u_hat_response` and `_u_hat_working` record two public residual domains;
- `_scores`, `_scores_response`, and `_scores_working` similarly coexist on
  different scales.

`Fepois` must copy the observation weights before delegating to this GLM path
because the inherited `_weights` field changes meaning. `Quantreg` does not
support fixed effects or weights, but it still reassigns formula-materialized
`_Y` and `_X` DataFrames to NumPy arrays before solving.

### Shared caches, result completion, and cleanup

For a multiple-estimation cache block, the runner gives each model a
`DemeanCache` backed by the same mutable dictionaries. Its linear-model cache
converts pandas inputs to arrays for demeaning, wraps the results in a
DataFrame, stores that frame, and converts selected columns back to arrays in
each model. The cache key is the retained-row index set because formulas in one
cache block share fixed effects and observation weights. GLM iterations do not
cache demeaned values because their working weights change; they share only a
preconditioner cache.

Model constructors also create many empty-array and `None` placeholders. The
runner then mutates the model through preparation, fitting, covariance
calculation, inference, performance statistics or IV first stages, and finally
`_clear_attributes()`. `store_data=False` deletes `_data`; `lean=True` deletes
the retained matrices and several fit products. The public `vcov()` method is
intentionally in-place and remains a post-fit mutation boundary. User input is
copied by default, with the documented `copy_data=False` path as the exception.

## Formula-state and lifecycle boundaries

`ModelMatrix` builds the formula inputs and is also the formula state a fitted
model retains. Missing, infinite, singleton, and other formula-level row filters
run during construction; afterwards the instance is treated as read-only, and
its dependent, independent, fixed-effect, IV, weight, and offset roles stay on
formula scale. Models populate legacy attributes from it in one
direction; later transformations produce separate within- or solver-scale
arrays and do not change the role or representation of the retained formula
state. Estimator-level filters that need the materialized design, such as GLM
separation, call `ModelMatrix.without_rows()`, which returns a filtered copy
whose `na_index` includes the dropped rows, so the canonical row sample and the
demeaning-cache key stay aligned with the data that enter IRLS.
`store_data=False` and `lean=True` discard the formula state together with the
other retained input state.

The generic runner operates on the structural `FittedModel` protocol. Response
validation, estimator-specific post-fit work, and result expansion live behind
model hooks rather than estimator-name dispatch in the runner. Pipeline-object
constructors only assemble configuration and child objects; for example,
`QuantregMulti` prepares its children in `prepare_model_matrix`, not during
construction.

This established the representation foundation. The within/weight cleanup
described below completes that layer by moving numerical primitives and
inference consumers to explicit within-scale inputs.

## Estimation-state vocabulary

New shared-core work should name the transformation domain instead of relying
on `_X`, `_Y`, `_Z`, or `_weights`. The immutable-state refactor uses
the following vocabulary:

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

The shared linear and GLM paths now implement the vocabulary above with frozen,
slotted state values. Frozen state prevents field rebinding, but contained NumPy
arrays remain mutable unless they are explicitly marked read-only:

| State | Persisted contract |
|---|---|
| `ModelMatrix` | Formula-materialized pandas tables remain on formula scale and keep dependent, independent, fixed-effect, IV, weight, and offset roles separate. |
| `ObservationWeights` | Canonical user-scale weights and their `aweights` or `fweights` semantics. `values=None` is the allocation-free unweighted path. |
| `WithinLinearData` | Unpremultiplied within-scale arrays, with response, design, instruments, and endogenous variables in named roles. |
| `GlmWorkingState` | Final within-scale working response and design, IRLS working weights, predictors, means, and response- and working-residual domains. |
| `DemeanedData` | Array-native cache entries whose ordered column names are metadata rather than DataFrame conversions around each reuse. |

Analytic weights keep the retained row count as the effective sample size;
frequency weights use the sum of their user-scale values. Probability weights
(`pweights`) remain unsupported. Fixed-effect projection may use observation or
IRLS weights, but the returned arrays remain within scale. OLS, IV, and IRLS
fit primitives create square-root-weighted design and response arrays only as
local solver temporaries. They persist response-unit residuals and weighted
scores or cross-products, not solver-scale copies of canonical data.
Singleton fixed-effect detection counts physical rows even under frequency
weights, so an aggregate row alone in its level is dropped although its literal
expansion would not be; this deliberate deviation is documented in
`tests/test_wls_types.py`.

GLMs keep two weight concepts deliberately separate. `ObservationWeights`
never changes after formula preparation, while each IRLS iteration computes
working weights and the final values live in `GlmWorkingState`. Response
residuals and working residuals likewise have separate fields.

The compatibility aliases are still available, but they are read-only
properties over the typed state rather than cross-type workspaces: the state
objects are the single writable representation, and assigning or deleting an
alias raises. For linear and IV fits, `_Y`, `_X`, and `_Z` view within-scale
arrays; for GLMs they view the final within-scale working response and design.
`_weights` always means observation weights, never square-root solver weights
or GLM working weights, and an unweighted fit materializes its ones column on
access instead of keeping one alive for the lifetime of the result. New code
should consume the typed state values rather than infer semantics from these
aliases.

`ccv()` explicitly rejects weighted models. `update()` can compute and return
coefficient-only Sherman-Morrison updates for unweighted, non-IV OLS without
fixed effects, but rejects `inplace=True`: design rows alone cannot reconstruct
the complete formula, prediction, inference, and performance state of a fitted
result.

Storage policy follows the full result graph. Retained IV first stages honor
`store_data=False`, and lean results discard within state, GLM working state,
demeaning caches, quantile solver outputs, and large first-stage arrays. Lean
results do keep the formula specification and evaluation context, so
`predict(newdata=...)` still works for models without fixed effects. Array-only
covariance updates remain available after `store_data=False`; cluster and HAC
updates require the estimation sample through the documented `vcov(data=...)`
argument. Explicit covariance data must already be filtered to the fitted row
sample and retain its estimation order. A row-count check rejects length
mismatches but cannot establish row identity or ordering. Lean results reject
post-fit covariance updates because their numerical arrays have been discarded.
Every other method that needs discarded state raises an informative error naming
the storage option and its remedy.

Keeping canonical arrays unpremultiplied favors readability without moving
weight work out of the numerical hot path: each solver still performs the same
vectorized square-root transform locally.

## Repository map and extension seams

Step-by-step recipes for estimators, post-estimation features, vcov types,
estimation-time options, and Rust kernels live in
[`AGENTS.md`](../../AGENTS.md) under "Wiring recipes". This table covers the
remaining seams.

| Change | Primary location | Pattern to follow |
|---|---|---|
| Model/result type | `estimation/models/<name>_.py` | nearest compatible result class |
| Shared numerical primitive | `estimation/internals/` | `fit_.py`, `vcov_.py`, or nearest analogue |
| Formula behavior | `estimation/formula/` | existing parser/model-matrix seams |
| DiD estimator | `pyfixest/did/` | nearest DiD API/result pair |
| User documentation | `docs/` | nearest tutorial/how-to plus quartodoc registration |

Public estimation functions use one module per entry point. Model modules end in
`_` so they do not shadow public functions. Compatibility shims in the
`estimation/` root are not implementation locations.

## Result and numerical boundaries

A model method validates inputs, unpacks model state, calls a module-level
function with keyword arguments, and stores or returns the result. Numerical
functions operate on arrays and return small typed dataclasses whose docstrings
state array shapes.

Every estimator or inference feature specifies behavior for weights, fixed
effects, IV, multiple estimation, `lean=True`, and `store_data=False`.
Unsupported combinations fail explicitly. Silent fallback is never acceptable.

## Compatibility changes

Fixest parity is the default. Record intentional differences in
[fixest-compatibility.md](fixest-compatibility.md) with their rationale and
tests. Public changes also require a changelog entry and, when compatibility
cannot be preserved directly, a reviewed deprecation path.
