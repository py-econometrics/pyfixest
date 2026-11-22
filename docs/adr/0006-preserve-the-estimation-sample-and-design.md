# Preserve the estimation sample and design across stages

Use the retained estimation sample and materialized design for downstream
calculations. Rebuilding either from the original inputs can change the row
set, stateful transformations, or fixed-effect codes and silently fit a
different model.

- Drop zero-weight observations before detecting singletons, as `fixest` does.
  Singleton detection operates on fixed effects without slopes. Integer and
  Boolean columns are finite by construction. Skip an empty frame drop to
  avoid rebuilding the frame. Frequency weights count repeated observations.
- Fixed-effect models use an intercept while materializing reduced-rank
  categorical encodings and remove it from the fitted design. A fixed-effects-only
  formula can request demeaning: it has no coefficient covariance and may
  have undefined fit statistics when residual degrees of freedom are absent.
- In IV, construct the first stage on the second stage's rows and with its
  options, including retention. Exclude endogenous regressors from the
  first-stage controls. Screen controls before instruments so a redundant
  instrument cannot displace a control. See the compatibility ledger for
  [first-stage tests](../developer/fixest-compatibility.md#first-stage-tests-with-redundant-instruments).
  Identification checks use the first-stage projection before second-stage
  collinearity screening. Check the endogenous column names first so an
  empty selection cannot masquerade as zero residual sum of squares.
  Apply the current absolute tolerance to the weighted first-stage residual
  sum of squares and to its change when adding the excluded instruments.
  Excluded instruments remain in instrument-matrix order.
- Refit against the retained sample, checking row identity before the runner
  resets the index. Reuse the preconditioner and skip singleton screening on
  the already-screened rows. The runner leaves the input frame intact.
  Separation also projects onto the original materialized design, and its
  iterative rectifier uses a tighter tolerance below zero to avoid false
  positives. Within residuals already account for the fixed-effect fit.
- Recover fitted values as response minus residual so the result includes the
  fixed-effect contribution. Under treatment normalization, omit the first
  observed level from the second and subsequent fixed effects. Those reference
  levels have zero coefficients, while unobserved levels remain missing. For
  small-sample corrections, only level intercepts can be nested in clusters.
  Count every varying-slope coefficient.
