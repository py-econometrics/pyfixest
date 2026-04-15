import warnings
from collections.abc import Callable
from typing import Any, TypedDict

from pyfixest.core.collinear import find_collinear_variables
from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.core.demean import demean, demean_within
from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all
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

# Try to import JAX functions, fall back to numba if not available
try:
    from pyfixest.estimation.jax.demean_jax_ import demean_jax as demean_jax_fn

    JAX_AVAILABLE = True
except ImportError:
    # Fall back to numba implementation if JAX is not available
    demean_jax_fn = demean_nb
    JAX_AVAILABLE = False

find_collinear_variables_jax = find_collinear_variables_nb
crv1_meat_loop_jax = crv1_meat_loop_nb
count_fixef_fully_nested_all_jax = count_fixef_fully_nested_all_nb

# Try to import CuPy functions, fall back to numba if not available
try:
    from pyfixest.estimation.cupy.demean_cupy_ import (
        demean_cupy32,
        demean_cupy64,
        demean_scipy,
    )

    CUPY_AVAILABLE = True
except ImportError:
    # Fall back to numba implementation if CuPy is not available
    demean_cupy32 = demean_nb
    demean_cupy64 = demean_nb
    demean_scipy = demean_nb
    CUPY_AVAILABLE = False

find_collinear_variables_cupy = find_collinear_variables_nb
crv1_meat_loop_cupy = crv1_meat_loop_nb
count_fixef_fully_nested_all_cupy = count_fixef_fully_nested_all_nb

# Try to import Torch functions, fall back to numba if not available
try:
    from pyfixest.estimation.torch.demean_torch_ import (
        demean_torch,
        demean_torch_cpu,
        demean_torch_cuda,
        demean_torch_cuda32,
        demean_torch_mps,
    )

    TORCH_AVAILABLE = True
except ImportError:
    demean_torch = demean_nb
    demean_torch_cpu = demean_nb
    demean_torch_mps = demean_nb
    demean_torch_cuda = demean_nb
    demean_torch_cuda32 = demean_nb
    TORCH_AVAILABLE = False

find_collinear_variables_torch = find_collinear_variables_nb
crv1_meat_loop_torch = crv1_meat_loop_nb
count_fixef_fully_nested_all_torch = count_fixef_fully_nested_all_nb

_TORCH_BACKEND_NAMES = frozenset(
    {"torch", "torch_cpu", "torch_mps", "torch_cuda", "torch_cuda32"}
)


class BackendFunctions(TypedDict):
    """Typed function bundle for a configured estimation backend."""

    demean: Callable[..., Any]
    collinear: Callable[..., Any]
    crv1_meat: Callable[..., Any]
    nonnested: Callable[..., Any]


BACKENDS: dict[str, BackendFunctions] = {
    "numba": {
        "demean": demean_nb,
        "collinear": find_collinear_variables_nb,
        "crv1_meat": crv1_meat_loop_nb,
        "nonnested": count_fixef_fully_nested_all_nb,
    },
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
    "jax": {
        "demean": demean_jax_fn,
        "collinear": find_collinear_variables_jax,
        "crv1_meat": crv1_meat_loop_jax,
        "nonnested": count_fixef_fully_nested_all_jax,
    },
    "cupy": {
        "demean": demean_cupy64,
        "collinear": find_collinear_variables_cupy,
        "crv1_meat": crv1_meat_loop_cupy,
        "nonnested": count_fixef_fully_nested_all_cupy,
    },
    "cupy32": {
        "demean": demean_cupy32,
        "collinear": find_collinear_variables_cupy,
        "crv1_meat": crv1_meat_loop_cupy,
        "nonnested": count_fixef_fully_nested_all_cupy,
    },
    "cupy64": {
        "demean": demean_cupy64,
        "collinear": find_collinear_variables_cupy,
        "crv1_meat": crv1_meat_loop_cupy,
        "nonnested": count_fixef_fully_nested_all_cupy,
    },
    "scipy": {
        "demean": demean_scipy,
        "collinear": find_collinear_variables_cupy,
        "crv1_meat": crv1_meat_loop_cupy,
        "nonnested": count_fixef_fully_nested_all_cupy,
    },
    **{
        name: {
            "demean": demean_fn,
            "collinear": find_collinear_variables_torch,
            "crv1_meat": crv1_meat_loop_torch,
            "nonnested": count_fixef_fully_nested_all_torch,
        }
        for name, demean_fn in [
            ("torch", demean_torch),
            ("torch_cpu", demean_torch_cpu),
            ("torch_mps", demean_torch_mps),
            ("torch_cuda", demean_torch_cuda),
            ("torch_cuda32", demean_torch_cuda32),
        ]
    },
}


def get_backend(demeaner_backend: str) -> BackendFunctions:
    """Resolve a backend implementation and warn for experimental torch paths."""
    try:
        impl = BACKENDS[demeaner_backend]
    except KeyError as exc:
        raise ValueError(f"Invalid demeaner backend: {demeaner_backend}") from exc

    if TORCH_AVAILABLE and demeaner_backend in _TORCH_BACKEND_NAMES:
        warnings.warn(
            (
                f"The `{demeaner_backend}` backend uses experimental torch "
                "algorithms. Behavior and performance may change in future "
                "releases."
            ),
            UserWarning,
            stacklevel=2,
        )

    return impl
