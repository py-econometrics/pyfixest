# Later calculations reuse the fit's estimation sample

A calculation that follows a fit, such as the IV first stage, a refit, or
fixed-effect recovery, starts from the estimation sample and model matrix the
fit retained. It does not rebuild them from the data the user passed.
Rebuilding can drop different rows and relearn stateful transformations and
fixed-effect codes, so the result would describe a different model than the
coefficients it is combined with. The cost is that these calculations need
retained state: when `lean` or `store_data` removed it, they raise a retention
error instead of recomputing.

## Consequences

- A refit on the estimation sample must keep every row. It skips singleton
  detection, reuses the fit's preconditioner, and raises if a row is dropped.
