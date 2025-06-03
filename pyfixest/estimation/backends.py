from pyfixest.core.collinear import find_collinear_variables
from pyfixest.core.crv1 import crv1_meat_loop
from pyfixest.core.demean import demean
from pyfixest.core.nested_fixed_effects import count_fixef_fully_nested_all
from pyfixest.estimation.demean_ import demean as demean_nb
from pyfixest.estimation.jax.demean_jax_ import demean_jax as demean_jax_fn
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb as find_collinear_variables_nb,
)
from pyfixest.estimation.numba.nested_fixef_nb import (
    _count_fixef_fully_nested_all as count_fixef_fully_nested_all_nb,
)
from pyfixest.estimation.vcov_utils import _crv1_meat_loop as crv1_meat_loop_nb

find_collinear_variables_jax = find_collinear_variables_nb
crv1_meat_loop_jax = crv1_meat_loop_nb
count_fixef_fully_nested_all_jax = count_fixef_fully_nested_all_nb

BACKENDS = {
    "numba": {
        "demean": demean_nb,
        "collinear": find_collinear_variables_nb,
        "crv1_meat": crv1_meat_loop_nb,
        "nested": count_fixef_fully_nested_all_nb,
    },
    "rust": {
        "demean": demean,
        "collinear": find_collinear_variables,
        "crv1_meat": crv1_meat_loop,
        "nested": count_fixef_fully_nested_all,
    },
    "jax": {
        "demean": demean_jax_fn,
        "collinear": find_collinear_variables_jax,
        "crv1_meat": crv1_meat_loop_jax,
        "nested": count_fixef_fully_nested_all_jax,
    },
}
