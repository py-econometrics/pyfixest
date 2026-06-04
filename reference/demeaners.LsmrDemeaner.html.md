# demeaners.LsmrDemeaner { #pyfixest.demeaners.LsmrDemeaner }

```python
demeaners.LsmrDemeaner(
    fixef_maxiter=1000,
    backend='within',
    precision='float64',
    device='auto',
    fixef_atol=1e-08,
    fixef_btol=1e-08,
    warn_on_cpu_fallback=True,
    preconditioner='auto',
    local_size=None,
)
```

Sparse LSMR demeaner.

## Notes {.doc-section .doc-section-notes}

The ``within`` backend takes a single tolerance, so ``fixef_atol`` and
``fixef_btol`` are collapsed to ``max(fixef_atol, fixef_btol)`` for that
backend. The ``cupy`` and ``torch`` backends use both tolerances
independently (SciPy LSMR convention).

The ``local_size`` field only applies to ``backend="within"``; the
``cupy`` and ``torch`` backends ignore it.

The ``precision``, ``device``, and ``warn_on_cpu_fallback`` fields are
only relevant for the ``torch`` backend (and the deprecated ``cupy``
backend). The ``within`` backend always runs on CPU in float64 and
ignores these fields.

``preconditioner`` selects the preconditioner. Supported values:

- ``"auto"`` (default): selects different preconditioners for different
  backend implementations: ``"schwarz"`` for ``within``; ``"diag"`` for
  ``torch`` / ``cupy``.
- ``"none"``: disables preconditioning. Supported by ``within`` and
  ``cupy``; not supported by ``torch``.
- ``"schwarz"``: additive Schwarz preconditioner. Only supported by the
  ``within`` backend.
- ``"diag"``: diagonal (Jacobi) preconditioner. Supported by ``within``,
  ``torch``, and ``cupy``.

If a value is incompatible with the chosen backend, a ``UserWarning`` is
emitted at solve time and the backend's default is used.