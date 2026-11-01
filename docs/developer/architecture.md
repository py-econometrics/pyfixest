# Pyfixest architecture

This file owns implementation conventions and shared contracts. Use the code
and its tests for method-level details; keep migration history in Git or the
relevant design discussion.

## Design principles

Follow the nearest implementation. Keep estimator-specific work in its own
modules, model methods responsible for orchestration, and numerical work in
standalone functions. Put measured, non-vectorizable hot loops in Rust; use
ordinary NumPy where it is clear and sufficient.

Prefer one authoritative representation of each value. A helper or dataclass
should make an operation or invariant easier to understand, eliminate actual
duplication, or own state with a clear lifetime. Avoid wrappers that only pass
through arguments, parallel access paths, and copies with no ownership reason.
Before removing one, inspect callers and subclass overrides: a short method
may be a necessary dispatch boundary.

## Stable core

The shared core owns formula parsing and model matrices, configuration and
multiple-estimation planning, demeaning and solver primitives, generic vcov
and inference, result interfaces, and backend contracts.

Changes to these contracts need maintainer design agreement, particularly
when expanding the core for one estimator. A request or accepted plan that
already specifies the contract change supplies that agreement. Implementation
changes preserving the contracts need no separate design approval. Resolve
only material undecided behavior before proceeding.

## Estimator add-ons

Start a new estimator as a standalone public function in `estimation/api/`
or its domain package, such as `did/`. Keep its preparation, iteration, and
post-processing in its own modules. Compose the existing formula, data,
demeaning, solver, vcov, and reporting primitives; do not add estimator-name
branches to generic runners merely for convenience.

Reuse a result class only when its semantics and supported operations fit.
Otherwise use a dedicated result with the applicable common accessors. Move a
helper into shared internals when it has a current shared consumer and a
generic contract; hypothetical reuse is insufficient.

## Estimation flow

The shared estimator APIs parse formulas and build `EstimationConfig`, with
typed `EstimationOptions` and `VcovSpec`. `plan_formulas` groups the parsed
formulas; it does not parse them again. `runner.run_estimation` and
`plan_.fit_one` drive the models through:

```text
configuration and model specs
  -> model-matrix preparation and response validation
  -> estimator-specific fitting
       OLS / IV: within arrays -> collinearity handling -> solve
       GLM: IRLS, separating observation and working weights
       quantreg: Frisch-Newton solve (no absorbed fixed effects)
  -> fit statistics -> covariance -> inference -> finalization
  -> retention cleanup -> fitted result / FixestMulti
```

DiD has domain-specific orchestration. `FixestMulti` is a container, not a
place for numerical computations. The generic runner uses the structural
`FittedModel` protocol; model hooks own response checks and estimator-specific
completion. Constructors assemble configuration and child objects, rather
than performing estimation.

## Formula-state and lifecycle boundaries

`ModelMatrix` retains formula-materialized tables, metadata, and the estimation
sample. Construction filters rows; later estimator-level filtering uses
`without_rows()` to return a filtered copy and update dropped-row tracking.
Keep row identity and cache keys aligned with the sample actually fitted.
Retained formula state is treated as read-only; within and solver transforms
produce separate arrays. Reuse fitted encoding and transform state for
prediction and refits when the operation requires the original design.

`DemeanCache` shares read-only demeaned arrays within a block of compatible
fixed effects and observation weights. The outer key records dropped rows;
column keys distinguish formula terms, not just their displayed names. GLM
working weights change each iteration, so GLMs reuse preconditioners but
recompute weighted projections. Read `internals/demean_.py` for the cache
implementation rather than duplicating its policy in new callers.

After fitting and inference, `_clear_attributes()` applies retention policy.
`lean=True` and `store_data=False` remove the prescribed state, including
formula state. `require_retained` raises `MissingModelDataError` for an
operation needing deliberately removed state. The public `vcov()` operation
is explicitly in-place; other operations must not leave a fitted result with
only part of its dependent state updated.

## Estimation-state vocabulary

Names should expose mathematical meaning, numerical scale, and ordering.

| Domain | Meaning |
|---|---|
| Formula data | Filtered, formula-materialized values and metadata; not necessarily raw input columns |
| Observation weights | User weights with their `aweights` or `fweights` interpretation |
| Within data | After the possibly weighted FE projection, in original units, not premultiplied by square-root weights |
| Solver data | Local arrays transformed for a numerical solve, such as `design_sqrt_weighted` |
| Working state | GLM iteration values: response, design, means, predictors, and working weights |
| Response residual | Residual in response units, distinct from a working or weighted solver residual |

Persisted fields must not change type or domain during fitting. Solver scratch
must not replace canonical within data. Use econometric names such as
`scores`, `meat`, `bread`, `u_hat`, and `clustid` where they match the quantity;
avoid aliases that make readers translate between names for the same thing.

## Implemented array and weight domains

The contracts below are defined in `internals/model_state.py` and the named
modules. Frozen dataclasses prevent field rebinding, not mutation of contained
NumPy arrays. Mark shared arrays read-only where the ownership contract needs
it; copy at actual ownership boundaries.

| Component | Contract |
|---|---|
| `ModelMatrix` | Formula-scale pandas tables with separate response, design, FE, IV, weight, and offset roles |
| `ObservationWeights` | User-scale weights and their semantics; `values=None` is the unweighted path |
| `WithinLinearData` / `WithinIvData` | Unpremultiplied within arrays; IV adds instruments and endogenous arrays |
| `GlmWorkingState` | Final IRLS values, with response and working residuals kept distinct |
| `SandwichComponents` | Weighted scores, Hessian, and bread; 2SLS scores use the first-stage projection |
| `DemeanedData` | Shared array-native cache entries and ordered column keys |

Analytic weights retain the row count as effective sample size; frequency
weights use the sum of user weights. `pweights` are unsupported. Singleton
detection counts physical rows even under frequency weights. Projection can
use observation or IRLS weights while returning within-scale arrays. Fit
primitives use square-root-weighted arrays locally and persist response-unit
residuals and weighted scores or cross-products. Observation weights stay
unchanged after preparation; IRLS working weights have separate state. Never
apply a weight twice
because its first application is hidden in another component.

## Repository map and extension seams

Paths below are relative to `pyfixest/` unless stated otherwise.

| Change | Location and precedent |
|---|---|
| Estimator API | `estimation/api/` or domain package; nearest API/result pair |
| Model/result | `estimation/models/<name>_.py`; nearest compatible result |
| Post-estimation | `estimation/post_estimation/`; e.g. `ritest.py` and its thin model wrapper |
| Numerical primitive | `estimation/internals/fit_.py`, `vcov_.py`, or nearest analogue |
| Formula behavior | `estimation/formula/`; existing parser/model-matrix seams |
| Orchestration | `estimation/config.py`, `plan_.py`, `runner.py`; typed options and model hooks |
| Demeaner configuration | `demeaners.py` |
| Native kernel | `core/` and repository-root `src/`; e.g. `src/nw.rs` and `core/nw.py` |
| DiD, reporting, utilities | `did/`, `report/`, `utils/` |
| Tests and user docs | Repository-root `tests/` and `docs/` |

Public estimation functions use one module per entry point. Model module names
end in `_` to avoid shadowing those functions. Root `estimation/` compatibility
shims are not implementation locations.

- **Vcov type:** alias in `internals/literals.py`, boundary parsing in
  `VcovSpec.from_user_input`, model support in `_check_vcov_support`, dispatch
  in `_vcov_from_spec`, math in `vcov_utils.py`, `vcov_.py`, or Rust, and
  multiple/quantile-estimation wiring where supported. Follow NW/DK HAC.
- **Estimation option:** shared typed alias, API validation, and a field in
  the applicable `EstimationOptions` type populated by the API.
- **Rust kernel:** `src/<topic>.rs`, registration in `src/lib.rs`, stub in
  `core/_core_impl.pyi`, and a wrapper in `core/`. Keep a readable NumPy
  reference where feasible.

Reuse existing context capture, dataframe conversion, cluster preparation,
`run_crv_loop`, and `_create_rng` instead of rederiving them.

## Result and numerical boundaries

Define and test each new estimator or inference feature's behavior for
`aweights`, `fweights`, FE, IV, multiple estimation, retention modes, and
relevant backends. For an existing feature, revisit the paths affected by the
change rather than producing a new matrix for unchanged contracts.

Declare post-estimation support in `Capabilities` and check it before reading
estimation state. Raise `NotImplementedError` naming unsupported combinations;
use retention errors for supported operations whose required state was removed.
Never reinterpret GLM working arrays or quantile output as linear-model state.
If a refit cannot replay the original estimation contract, reject that path.
If an operation cannot reconstruct a complete fitted result, return its value
instead of mutating the result in place.

Model methods validate, unpack state, call numerical functions with keyword
arguments, and store or return results. Numerical functions take arrays and
return small typed results with documented shapes. Keep the algorithm legible;
splitting a solver loop must not obscure the math or harm compilation.

## Code and documentation conventions

- Use `from __future__ import annotations`, PEP 604 unions, keyword arguments
  for internal calls, and `NDArray[np.float64]` in stubs. Shared typed option
  aliases belong in `estimation/internals/literals.py`.
- Validate options at the API boundary with `ValueError` naming allowed values;
  use the flat `pyfixest/errors/` classes for domain failures. Errors should
  identify the rejected input and, when useful, a supported alternative.
- Guard optional imports; raise an actionable error naming the pip extra when
  the missing dependency's path is used.
- Use `np.random.default_rng(seed)`, never global seeding. Never mutate user
  input except through the documented `copy_data=False` path.
- Internal docstrings start with what the function does. Add shapes, units,
  ordering, ownership, and mathematical explanation only where they clarify
  a non-obvious contract. Cite the method's paper at its numerical implementation.
  Do not turn a convenience helper into a tutorial or retain migration narration.

## Public documentation

User-facing functions, methods, and classes need NumPy docstrings with complete
Parameters/Returns, executable `{python}` examples, root-relative `.qmd` links,
and linked papers for econometric methods. New public entries need exports and
quartodoc registration. Being described in the reference does not by itself
make an internal constructor a supported user entry point; explain how users
obtain and use a result object.

Describe behavior plainly. Keep routine “as in fixest” assurances in tests and
the compatibility ledger, and explain a difference to users when it affects
their choices. Extend the nearest user guide when a workflow needs explanation;
do not require a new vignette for each small change.

Add a concise changelog entry for features, behavior/default changes, bug
fixes, deprecations, performance changes, or new contributor tooling. Internal
refactors, guidance edits, CI tweaks, and typo fixes need no entry. Keep
entries about what users or contributors need to know.

## Compatibility changes

Use the [compatibility policy](fixest-compatibility.md) for intentional
differences and their evidence. Preserve stable public behavior or use an
agreed deprecation path; do not infer permission to break it from a cleanup.
