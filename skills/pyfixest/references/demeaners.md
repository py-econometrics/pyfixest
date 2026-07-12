# Demeaners

Use the default MapDemeaner() for typical high-dimensional fixed effects. Choose
MapDemeaner(backend=numba) only when its optional dependency is available. Use
LsmrDemeaner() for sparse or difficult fixed-effects designs; the optional torch
backend needs the corresponding extra and device support.

Pass a typed demeaner= configuration rather than deprecated loose backend or
tolerance arguments. The result exposes a reusable preconditioner for a later
model with the same fixed-effect design; do not reuse it across a different
design.

Read pyfixest/docs/pages/how-to/demeaner-backends.md for backend limits,
tolerances, preconditioners, and optional-dependency setup.
