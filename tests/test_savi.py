import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.post_estimation.savi import _savi_e_value
from pyfixest.utils.utils import ssc


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


def _longley_data():
    return pd.DataFrame(
        {
            "TOTEMP": [
                60.323,
                61.122,
                60.171,
                61.187,
                63.221,
                63.639,
                64.989,
                63.761,
                66.019,
                67.857,
                68.169,
                66.513,
                68.655,
                69.564,
                69.331,
                70.551,
            ],
            "GNP": [
                234.289,
                259.426,
                258.054,
                284.599,
                328.975,
                346.999,
                365.385,
                363.112,
                397.469,
                419.180,
                442.769,
                444.546,
                482.704,
                502.601,
                518.173,
                554.894,
            ],
            "UNEMP": [
                235.6,
                232.5,
                368.2,
                335.1,
                209.9,
                193.2,
                187.0,
                357.8,
                290.4,
                282.2,
                293.6,
                468.1,
                381.3,
                393.1,
                480.6,
                400.7,
            ],
        }
    )


def test_savi_e_value_matches_gaussian_large_sample_limit():
    dfn = 1
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


@pytest.mark.parametrize("vcov", ["iid", "hetero"])
@pytest.mark.parametrize("mixture_precision", [1.0, 2.5])
def test_savi_confidence_sequences_contain_confidence_intervals(
    vcov, mixture_precision
):
    data = pf.get_data()
    fit = feols("Y ~ X1 + X2", data, vcov=vcov)
    alpha = 0.05
    cs = fit.confint(
        alpha=alpha,
        inference_type="savi",
        mixture_precision=mixture_precision,
    )
    ci = fit.confint(alpha=alpha)

    assert np.all(cs["2.5%"] <= ci["2.5%"])
    assert np.all(cs["97.5%"] >= ci["97.5%"])


@pytest.mark.parametrize("vcov", ["iid", "hetero", "HC1", "HC2", "HC3"])
def test_savi_supports_iid_and_hc_vcov(vcov):
    fit = feols("Y ~ X1 + X2", pf.get_data(), vcov=vcov)

    fit.evalue()


def test_savi_matches_avlm_longley_iid():
    fit = feols("TOTEMP ~ GNP + UNEMP", _longley_data())
    mixture_precision = 2.5

    avlm_pvalues = np.array(
        [
            2.3994985366968640e-06,
            6.5452796798514853e-06,
            1.3043518743909294e-01,
        ]
    )
    avlm_conf_int = pd.DataFrame(
        {
            "2.5%": [
                50.301901821972919038,
                0.031635798521018964,
                -0.012035233090925803,
            ],
            "97.5%": [
                54.4624322783199232845,
                0.0440448555138813133,
                0.0011637464493843685,
            ],
        },
        index=fit.coef().index,
    )
    np.testing.assert_allclose(
        fit.evalue(mixture_precision=mixture_precision),
        1 / avlm_pvalues,
    )
    np.testing.assert_allclose(
        fit.pvalue_savi(mixture_precision=mixture_precision),
        avlm_pvalues,
    )
    pd.testing.assert_frame_equal(
        fit.confint(
            inference_type="savi",
            mixture_precision=mixture_precision,
        ),
        avlm_conf_int,
        check_names=False,
    )


def test_savi_hc0_coefficients_match_avlm_longley():
    fit = feols(
        "TOTEMP ~ GNP + UNEMP",
        _longley_data(),
        vcov="hetero",
        ssc=ssc(k_adj=False),
    )
    mixture_precision = 2.5

    avlm_pvalues = np.array(
        [
            2.3258728655056687e-06,
            4.3782614065014952e-06,
            4.8523079107576070e-02,
        ]
    )
    avlm_conf_int = pd.DataFrame(
        {
            "2.5%": [
                50.840615331130983634,
                0.033030806719240489,
                -0.010841212821887157,
            ],
            "97.5%": [
                53.923718769161859,
                0.042649847315659788,
                -0.000030273819654277542,
            ],
        },
        index=fit.coef().index,
    )

    np.testing.assert_allclose(
        fit.evalue(mixture_precision=mixture_precision),
        1 / avlm_pvalues,
    )
    np.testing.assert_allclose(
        fit.pvalue_savi(mixture_precision=mixture_precision),
        avlm_pvalues,
    )
    pd.testing.assert_frame_equal(
        fit.confint(
            inference_type="savi",
            mixture_precision=mixture_precision,
        ),
        avlm_conf_int,
        check_names=False,
    )


def test_optimal_mixture_precision_is_a_local_minimum_of_the_confidence_radius():
    nobs = 10_000
    number_of_coefficients = 5
    alpha = 0.05
    dfd = nobs - number_of_coefficients

    def radius(g):
        return _confidence_radius(alpha=alpha, mixture_precision=g, nobs=nobs, dfd=dfd)

    g_star = pf.optimal_mixture_precision(nobs, number_of_coefficients, alpha)
    f_min = radius(g_star)

    delta = max(g_star * 0.01, 1e-4)
    grid = np.linspace(max(1.0, g_star - 5 * delta), g_star + 5 * delta, 25)
    grid_values = np.array([radius(g) for g in grid])

    assert np.all(grid_values >= f_min - 1e-9)
