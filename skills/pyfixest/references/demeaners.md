# Demeaning backends

The `demeaner=` argument controls how high-dimensional fixed effects are removed.
Read the bundled
[demeaner guide](../../../pyfixest/docs/pages/how-to/demeaner-backends.md) and
[difficult-fixed-effects explanation](../../../pyfixest/docs/pages/explanation/difficult-fixed-effects.md)
before overriding the default.

- `MapDemeaner` is the default and is appropriate for most problems.
- `LsmrDemeaner` is useful for difficult or sparse fixed-effect structures.
- Optional Numba and Torch backends have extra dependency and hardware
  requirements; verify availability before selecting them.

Start with the default, inspect convergence diagnostics or errors, and change
the backend only for a concrete numerical or performance reason. A reusable
preconditioner is valid only when the fixed-effect structure is unchanged.
Treat tolerances, iteration limits, and backend selection as numerical choices
that should be reported when they affect reproducibility.
