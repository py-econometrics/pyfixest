# Preconditioner

``` python
Preconditioner(data, build_time_seconds)
```

Opaque handle to a pre-built within preconditioner (Additive Schwarz or Diagonal Jacobi).

Equality / hashing follow Python’s pyo3 defaults (object identity), in line with upstream `within._within.Preconditioner`. Pickle uses `postcard` round-tripping via `__reduce__`.
