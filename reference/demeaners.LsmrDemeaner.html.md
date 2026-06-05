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
  backend implementations: ``"additive"`` for ``within``; ``"diagonal"``
  for ``torch`` / ``cupy``.
- ``"off"``: disables preconditioning. Supported by ``within`` and
  ``cupy``; not supported by ``torch``.
- ``"additive"``: additive Schwarz preconditioner. Only supported by the
  ``within`` backend.
- ``"diagonal"``: diagonal (Jacobi) preconditioner. Supported by
  ``within``, ``torch``, and ``cupy``.
- A :class:`pyfixest.Preconditioner` instance: a previously built
  preconditioner (typically obtained via ``fit.preconditioner`` or
  pickled across sessions). Only supported by ``backend='within'``;
  preconditioners are only computed and applied for two or more
  fixed-effect factors because single-factor problems run MAP as the within algo
  provides no benefits. Passing a preconditioner to any other backend raises ``ValueError``
  at construction time.

If a *string* value is incompatible with the chosen backend, a
``UserWarning`` is emitted at solve time and the backend's default is
used. A ``Preconditioner`` paired with a non-``within`` backend is
rejected eagerly with ``ValueError`` because there is no sensible
fallback for a prebuilt object.