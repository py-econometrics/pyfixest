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
