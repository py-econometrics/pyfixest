import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.post_estimation.savi import (
    _savi_e_value,
    optimal_mixture_precision,
)
from pyfixest.estimation.post_estimation.wald import _wald_statistic
from pyfixest.utils.utils import ssc


def _gaussian_e_value(qvalue, dfn, nobs, mixture_precision):
    g_ratio = mixture_precision / (mixture_precision + nobs)
    log_e_value = (
        dfn / 2 * np.log(g_ratio) + 0.5 * nobs / (mixture_precision + nobs) * qvalue
    )
    return np.exp(log_e_value)


def _e_value_from_f_statistic(f_statistic, dfn, dfd, nobs, mixture_precision):
    g_ratio = mixture_precision / (mixture_precision + nobs)
    stat_ratio = dfn / dfd * f_statistic
    return np.exp(
        dfn / 2 * np.log(g_ratio)
        + (dfd + dfn) / 2 * (np.log1p(stat_ratio) - np.log1p(g_ratio * stat_ratio))
    )


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


def test_savi_default_f_test_matches_explicit_identity_restriction():
    fit = feols("Y ~ X1 + X2", pf.get_data())
    mixture_precision = 2.5

    default_f_test = fit.wald_test()
    default_e_value = _e_value_from_f_statistic(
        default_f_test["statistic"],
        dfn=fit._dfn,
        dfd=fit._df_t,
        nobs=fit._N,
        mixture_precision=mixture_precision,
    )

    np.testing.assert_allclose(
        fit.evalue(R=np.eye(fit._k), mixture_precision=mixture_precision),
        default_e_value,
    )


def test_savi_joint_restriction_with_nonzero_q():
    fit = feols("Y ~ X1 + X2", pf.get_data())
    restriction = np.array([[0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    q = np.array([0.5, -0.25])
    mixture_precision = 2.5

    distance = restriction @ fit._beta_hat - q
    restricted_vcov = restriction @ fit._vcov @ restriction.T
    wald_statistic = distance.T @ np.linalg.pinv(restricted_vcov) @ distance
    expected = _e_value_from_f_statistic(
        wald_statistic / restriction.shape[0],
        dfn=restriction.shape[0],
        dfd=fit._df_t,
        nobs=fit._N,
        mixture_precision=mixture_precision,
    )

    np.testing.assert_allclose(
        fit.evalue(R=restriction, q=q, mixture_precision=mixture_precision),
        expected,
    )


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
    radius = _confidence_radius(
        alpha=alpha,
        mixture_precision=mixture_precision,
        nobs=fit._N,
        dfd=fit._df_t,
    )
    expected = pd.DataFrame(
        {
            "2.5%": fit.coef() - np.sqrt(radius) * fit.se(),
            "97.5%": fit.coef() + np.sqrt(radius) * fit.se(),
        }
    )

    pd.testing.assert_frame_equal(cs, expected, check_names=False)
    assert np.all(cs["2.5%"] <= ci["2.5%"])
    assert np.all(cs["97.5%"] >= ci["97.5%"])


@pytest.mark.parametrize("vcov", ["iid", "hetero", "HC1", "HC2", "HC3"])
def test_savi_supports_iid_and_hc_vcov(vcov):
    fit = feols("Y ~ X1 + X2", pf.get_data(), vcov=vcov)

    fit.evalue()


def test_savi_tidy_and_summary_replace_p_values(capsys):
    fit = feols("Y ~ X1 + X2", pf.get_data())
    tidy_regular = fit.tidy()
    tidy_savi = fit.tidy(inference_type="savi", mixture_precision=2.5)

    assert "Pr(>|t|)" in tidy_regular.columns
    assert "Pr(>|t|)" not in tidy_savi.columns
    assert "e_value" in tidy_savi.columns

    fit.summary(inference_type="savi", mixture_precision=2.5)
    out = capsys.readouterr().out
    assert "e_value" in out
    assert "Inference:  iid (savi)" in out


def test_savi_tidy_preserves_sample_split_column():
    fits = feols("Y ~ X1 + X2", pf.get_data(), split="f1")

    for fit in fits.to_list():
        regular = fit.tidy()
        savi = fit.tidy(inference_type="savi")

        pd.testing.assert_series_equal(savi["Sample"], regular["Sample"])


def test_savi_matches_avlm_and_statsmodels_longley_iid():
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
        fit.pvalue(inference_type="savi", mixture_precision=mixture_precision),
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


def test_savi_f_test_matches_avlm_longley_iid():
    fit = feols("TOTEMP ~ GNP + UNEMP", _longley_data())
    restriction = np.eye(fit._k)[1:]
    mixture_precision = 2.5

    wald_statistic, dfn = _wald_statistic(
        beta_hat=fit._beta_hat,
        vcov=fit._vcov,
        R=restriction,
    )
    f_statistic = wald_statistic / dfn

    # summary(avlm::av(lm(Employed ~ GNP + Unemployed, longley)))$fstatistic
    avlm_f_statistic = 329.49763608456516
    avlm_f_pvalue = 5.3726884577847966e-06
    assert dfn == 2
    np.testing.assert_allclose(f_statistic, avlm_f_statistic, rtol=1e-12)
    np.testing.assert_allclose(
        fit.evalue(R=restriction, mixture_precision=mixture_precision),
        1 / avlm_f_pvalue,
    )
    np.testing.assert_allclose(
        fit.sequential_pvalue(
            R=restriction,
            mixture_precision=mixture_precision,
        ),
        avlm_f_pvalue,
    )


def test_savi_hc0_coefficients_match_avlm_and_statsmodels_longley():
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
        fit.pvalue(inference_type="savi", mixture_precision=mixture_precision),
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

    g_star = optimal_mixture_precision(nobs, number_of_coefficients, alpha)
    f_min = radius(g_star)

    delta = max(g_star * 0.01, 1e-4)
    grid = np.linspace(max(1.0, g_star - 5 * delta), g_star + 5 * delta, 25)
    grid_values = np.array([radius(g) for g in grid])

    assert np.all(grid_values >= f_min - 1e-9)
