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

The accepted decisions behind these principles are in
[developer/decisions](decisions/).

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

Each public estimator builds an `EstimationConfig`. `parse_formula` expands
multiple-estimation syntax, and `runner.run_estimation` and `fit_one` prepare
each model before dispatching to its estimator-specific fit path:

```text
public API
  -> EstimationConfig
  -> parse_formula
  -> prepare_model_matrix
  -> estimator-specific get_fit
       -> feols / feiv
            -> demean
            -> to_array
            -> drop_multicol_vars
            -> wls_transform
            -> solve OLS / 2SLS
       -> fepois / feglm
            -> IRLS with weighted demeaning and solving inside each iteration
       -> quantreg
            -> Frisch-Newton solve (absorbed fixed effects are unsupported)
  -> vcov
  -> get_inference
  -> fitted result / FixestMulti
```

`FixestMulti` is a container for fitted results. Numerical behavior belongs in
the individual models and shared primitives, not in the container.

## Repository map and extension seams

| Change | Primary location | Pattern to follow |
|---|---|---|
| Estimation entry point | `estimation/api/<name>.py` | `api/quantreg.py` and its export chain |
| Model/result type | `estimation/models/<name>_.py` | nearest compatible result class |
| Post-estimation operation | `estimation/post_estimation/` | `ritest.py` plus thin model wrapper |
| Shared numerical primitive | `estimation/internals/` | `fit_.py`, `vcov_.py`, or nearest analogue |
| Formula behavior | `estimation/formula/` | existing parser/model-matrix seams |
| Native hot loop | `src/` and `pyfixest/core/` | `src/nw.rs` → `core/nw.py` |
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
