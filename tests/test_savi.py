import numpy as np
import pytest

from pyfixest.estimation.post_estimation.savi import (
    _savi_e_value,
    optimal_mixture_precision,
)


def _gaussian_e_value(qvalue, dfn, nobs, mixture_precision):
    g_ratio = mixture_precision / (mixture_precision + nobs)
    log_e_value = (
        dfn / 2 * np.log(g_ratio) + 0.5 * nobs / (mixture_precision + nobs) * qvalue
    )
    return np.exp(log_e_value)


def _confidence_radius(alpha, mixture_precision, nobs, dfd):
    g_ratio = mixture_precision / (mixture_precision + nobs)
    boundary = (alpha**2 * g_ratio) ** (1 / (dfd + 1))
    denominator = boundary - g_ratio
    if denominator <= 0:
        return np.inf
    return dfd * (1 - boundary) / denominator


@pytest.mark.parametrize("dfn", [1, 3])
def test_savi_e_values_converge_to_gaussian_e_process(dfn):
    mixture_precision = 2.5
    qvalue = 8.0
    f_statistic = qvalue / dfn
    nobs = 10_000_000
    dfd = nobs - 10

    e_value = _savi_e_value(
        f_statistic,
        dfn=dfn,
        dfd=dfd,
        nobs=nobs,
        mixture_precision=mixture_precision,
    )
    gaussian_e_value = _gaussian_e_value(qvalue, dfn, nobs, mixture_precision)
    np.testing.assert_allclose(e_value, gaussian_e_value, rtol=1e-5)


def test_optimal_mixture_precision_is_a_local_minimum_of_the_confidence_radius():
    nobs = 10_000
    number_of_coefficients = 5
    alpha = 0.05
    dfd = nobs - number_of_coefficients

    def radius(g):
        return _confidence_radius(alpha=alpha, mixture_precision=g, nobs=nobs, dfd=dfd)

    g_star = optimal_mixture_precision(nobs, number_of_coefficients, alpha)
    f_min = radius(g_star)

    delta = max(g_star * 0.01, 1e-4)
    grid = np.linspace(max(1.0, g_star - 5 * delta), g_star + 5 * delta, 25)
    grid_values = np.array([radius(g) for g in grid])

    assert np.all(grid_values >= f_min - 1e-9)
