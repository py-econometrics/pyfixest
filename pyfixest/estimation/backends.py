from pyfixest.core.collinear import find_collinear_variables
from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.core.demean import demean
from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all
from pyfixest.estimation.demean_ import demean as demean_nb
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb as find_collinear_variables_nb,
)
from pyfixest.estimation.numba.nested_fixef_nb import (
    _count_fixef_fully_nested_all as count_fixef_fully_nested_all_nb,
)
from pyfixest.estimation.vcov_utils import _crv1_meat_loop as crv1_meat_loop_nb

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
    )

    CUPY_AVAILABLE = True
except ImportError:
    # Fall back to numba implementation if CuPy is not available
    demean_cupy32 = demean_nb
    demean_cupy64 = demean_nb
    CUPY_AVAILABLE = False

find_collinear_variables_cupy = find_collinear_variables_nb
crv1_meat_loop_cupy = crv1_meat_loop_nb
count_fixef_fully_nested_all_cupy = count_fixef_fully_nested_all_nb

BACKENDS = {
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
}