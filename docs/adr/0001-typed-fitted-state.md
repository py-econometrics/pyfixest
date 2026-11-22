# Fitted state lives in typed frozen objects

A fitted model publishes its estimation state as typed frozen dataclasses,
such as `fit.sample_info`, `fit.variance_covariance`, `fit.coeftable`,
`fit.sandwich`, and `fit.fitstat`. It does not publish that state as loose
private attributes that methods set one at a time. Each value then has one
authoritative representation, and an operation either replaces a whole
component or leaves it alone, so a fit cannot be left with only part of its
dependent state updated.

## Consequences

- Freezing prevents field rebinding, not mutation of contained NumPy arrays.
  Mark shared arrays read-only where the ownership contract needs it.
- External code that read the 0.60 private attributes broke. #2028 keeps
  aliases only for the names released integrations (maketables,
  marginaleffects) and the 0.60 documentation used.

## Ownership and replay

The demeaning cache keeps responses ahead of regressors so it can hand out
contiguous column views when available. Both those views and copies produced
by scattered indexing are read-only. The caller's ability to mutate a result
must be independent of the order in which the cache filled. Unweighted Poisson
fits use the shared `None` representation to avoid allocating unit weights.
Weight constructors validate their state at publication, and post-estimation
operations make local copies before applying weights in place. The coefficient
names track the retained design columns even when the stored model matrix still
includes columns removed for collinearity.

A quantile process carries the first requested quantile in its options. Each
child fit replaces that quantile and inherits the other options.

Legacy attribute descriptors preserve the migration message when a descriptor
raises `AttributeError`. The fallback enumerates known names so the type checker
can still catch unknown attributes. The fallback may also replace an error
from an unrelated property, so properties should avoid raising
`AttributeError` internally. Unpublished or discarded state is absent for
retention checks and `hasattr`. The linear and quantile deviance aliases retain
the `None` value exposed by 0.60.
