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
