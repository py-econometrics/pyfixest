# Post-estimation leaves a fit's inference intact

A post-estimation operation that needs a different variance estimator than the
fitted model's, such as a cluster variable the fit did not use, computes it on
a local refit. It leaves `fit.variance_covariance` and the coefficient table
unchanged. Causal cluster variance and randomization inference used to call
`fit.vcov()` and warn that the stored inference was replaced, so running a
test changed the standard errors the fit reported afterwards (#1988, #1989).
Replacing a fit's inference remains the user's explicit `vcov()` call.

## Consequences

- The local refit keeps the fit's small-sample correction, so the operation's
  statistics stay comparable with the fit's.
- An operation that recomputes the fit's test statistic on other data, such
  as the permuted statistics of randomization inference, uses the fit's
  variance estimator and small-sample correction.
