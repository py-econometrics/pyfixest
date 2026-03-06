from pyfixest.core.collinear import find_collinear_variables
from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.core.demean import demean, demean_within
from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all


def _lazy_numba():
    """Lazily import numba backends only when needed."""
    from pyfixest.estimation.internals.demean_ import demean as demean_nb
    from pyfixest.estimation.internals.vcov_utils import (
        _crv1_meat_loop as crv1_meat_loop_nb,
    )
    from pyfixest.estimation.numba.find_collinear_variables_nb import (
        _find_collinear_variables_nb as find_collinear_variables_nb,
    )
    from pyfixest.estimation.numba.nested_fixef_nb import (
        _count_fixef_fully_nested_all as count_fixef_fully_nested_all_nb,
    )
    return {
        "demean": demean_nb,
        "collinear": find_collinear_variables_nb,
        "crv1_meat": crv1_meat_loop_nb,
        "nonnested": count_fixef_fully_nested_all_nb,
    }


def _lazy_jax():
    """Lazily import JAX backends only when needed."""
    try:
        from pyfixest.estimation.jax.demean_jax_ import demean_jax as demean_jax_fn
    except ImportError:
        nb = _lazy_numba()
        return nb
    nb = _lazy_numba()
    return {
        "demean": demean_jax_fn,
        "collinear": nb["collinear"],
        "crv1_meat": nb["crv1_meat"],
        "nonnested": nb["nonnested"],
    }


def _lazy_cupy(precision="64"):
    """Lazily import CuPy backends only when needed."""
    try:
        from pyfixest.estimation.cupy.demean_cupy_ import (
            demean_cupy32,
            demean_cupy64,
            demean_scipy,
        )
        CUPY_AVAILABLE = True
    except ImportError:
        CUPY_AVAILABLE = False

    nb = _lazy_numba()

    if not CUPY_AVAILABLE:
        return nb

    demean_fn = {"32": demean_cupy32, "64": demean_cupy64, "scipy": demean_scipy}.get(
        precision, demean_cupy64
    )
    return {
        "demean": demean_fn,
        "collinear": nb["collinear"],
        "crv1_meat": nb["crv1_meat"],
        "nonnested": nb["nonnested"],
    }


class _LazyBackends(dict):
    """Dict that lazily loads numba/jax/cupy backends on first access."""

    _EAGER = {
        "rust": {
            "demean": demean,
            "collinear": find_collinear_variables,
            "crv1_meat": crv1_meat_loop,
            "nonnested": count_fixef_fully_nested_all,
        },
        "rust-cg": {
            "demean": demean_within,
            "collinear": find_collinear_variables,
            "crv1_meat": crv1_meat_loop,
            "nonnested": count_fixef_fully_nested_all,
        },
    }

    _LAZY_LOADERS = {
        "numba": lambda: _lazy_numba(),
        "jax": lambda: _lazy_jax(),
        "cupy": lambda: _lazy_cupy("64"),
        "cupy32": lambda: _lazy_cupy("32"),
        "cupy64": lambda: _lazy_cupy("64"),
        "scipy": lambda: _lazy_cupy("scipy"),
    }

    def __init__(self):
        super().__init__(self._EAGER)

    def __getitem__(self, key):
        if key not in self:
            loader = self._LAZY_LOADERS.get(key)
            if loader is None:
                raise KeyError(f"Unknown backend: {key}")
            self[key] = loader()
        return super().__getitem__(key)

    def __contains__(self, key):
        return super().__contains__(key) or key in self._LAZY_LOADERS


BACKENDS = _LazyBackends()
