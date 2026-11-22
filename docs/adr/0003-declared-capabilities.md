# Post-estimation support is declared, not inferred

Each model class declares one `Capabilities` value, which lists the inference
and post-estimation operations it supports. An operation checks that
declaration before it reads any estimation state. The alternative,
`isinstance` and model-name checks inside each post-estimation method, spread
one estimator's limits across many files and let unsupported paths
reinterpret another estimator's arrays, such as GLM working arrays read as
linear-model state (#1728).

## Consequences

- An unsupported combination raises `NotImplementedError` naming it. A
  supported operation whose state was removed by retention raises a retention
  error instead.

## Estimator-specific constraints

Fixed-effect recovery and related operations depend on the retained codes.
Quantile regression rejects weights and fixed effects and skips the shared
demeaning cache. Event-study results disable randomization inference, update,
and SAVI. Generic GLMs cannot use the inherited slow CRV3 jackknife until its
refits preserve the GLM family. The fast CRV3 and randomization-inference
algorithms rely on a single least-squares solve. Two-stage IV, iterative GLM
fits, and quantile loss each have a different fitting procedure. Separation
checks apply to families with outcome bounds.
