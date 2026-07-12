<!-- Generated from docs/reference/core.demean.Preconditioner.qmd; do not edit. -->

# core.demean.Preconditioner

```python
core.demean.Preconditioner()
```

Opaque handle to a pre-built within preconditioner (Additive Schwarz or
Diagonal Jacobi).

Equality / hashing follow Python's pyo3 defaults (object identity), in
line with upstream ``within._within.Preconditioner``. Pickle uses
``postcard`` round-tripping via ``__reduce__``.

## Attributes

| Name | Description |
| --- | --- |
| [build_time_seconds](#pyfixest.core.demean.Preconditioner.build_time_seconds) |  |
| [ncols](#pyfixest.core.demean.Preconditioner.ncols) |  |
| [nrows](#pyfixest.core.demean.Preconditioner.nrows) |  |
| [variant](#pyfixest.core.demean.Preconditioner.variant) |  |
