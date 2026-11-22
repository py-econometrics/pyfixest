# New estimators are add-ons to a stable core

A new estimator starts as a standalone estimation function in its own modules.
It composes the shared formula, demeaning, solver, covariance, and reporting
primitives, and adds no estimator-name branches to the generic runners or to
another estimator's model class. Subclassing `Feols` and branching on the model
name was the alternative. We rejected it because every estimator's special
cases leaked into shared code. Subclass hooks have since replaced those
branches in `Feols` (#1616).

## Consequences

- An estimator reuses a result class only when its semantics and supported
  operations fit. Otherwise it gets a dedicated result.
- A helper moves into the shared core when it has a current second consumer.
  Hypothetical reuse is not enough.

## Import boundaries

The top-level package eagerly exposes version information and resolves its
public functions and submodules on demand. Dedicated API modules can load
directly. The slow randomization-inference path works without Numba, so only
the fast path imports it and reports a missing dependency.

Capture the caller's formula context at the public wrapper before dispatch.
Forwarding an integer frame offset through `fepois` to `feglm` would resolve it
relative to the new frame. Keep shared dispatch types in the dependency-light
literals module, use annotation-only imports in utilities imported by
estimation, and defer the refit module's import of the runner to avoid a cycle.
