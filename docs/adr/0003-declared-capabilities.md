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
