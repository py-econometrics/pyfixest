import pyfixest_core
from pyfixest.estimation.demean_ import demean as demean_nb
from pyfixest.estimation.jax.demean_jax_ import demean_jax as demean_jax_fn
from pyfixest.estimation.numba.find_collinear_variables_nb import (
    _find_collinear_variables_nb as find_collinear_variables_nb,
)
from pyfixest.estimation.numba.nested_fixef_nb import (
    _count_fixef_fully_nested_all as count_fixef_fully_nested_all_nb,
)
from pyfixest.estimation.vcov_utils import _crv1_meat_loop as crv1_meat_loop_nb
from pyfixest.estimation.rust.demean_rs_ import demean_rs

find_collinear_variables_rs = pyfixest_core.find_collinear_variables_rs
crv1_meat_loop_rs = pyfixest_core.crv1_meat_loop_rs
count_fixef_fully_nested_all_rs = pyfixest_core.count_fixef_fully_nested_all_rs

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
        "demean": demean_rs,
        "collinear": find_collinear_variables_rs,
        "crv1_meat": crv1_meat_loop_rs,
        "nested": count_fixef_fully_nested_all_rs,
    },
    "jax": {
        "demean": demean_jax_fn,
        "collinear": find_collinear_variables_jax,
        "crv1_meat": crv1_meat_loop_jax,
        "nested": count_fixef_fully_nested_all_jax,
    },
}
