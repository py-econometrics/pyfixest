import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import pyfixest as pf
from pyfixest.estimation import feols
from pyfixest.estimation.post_estimation.savi import (
    _savi_cluster_e_value,
    _savi_e_value,
)
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


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "group_id"}])
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

    # Test against hard-coded results from avlm.
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
    """Check that the closed-form precision minimizes confidence-sequence width."""
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


def _hotelling_e_value(t_squared, n_units, dfn, mixture_precision):
    g_ratio = mixture_precision / (mixture_precision + n_units)
    return g_ratio ** (dfn / 2) * (
        (1 + t_squared / (n_units - 1)) / (1 + g_ratio * t_squared / (n_units - 1))
    ) ** (n_units / 2)


def _hotelling_radius(alpha, mixture_precision, n_units):
    g_ratio = mixture_precision / (mixture_precision + n_units)
    boundary = (alpha**2 * g_ratio) ** (1 / n_units)
    return np.sqrt((n_units - 1) * (1 - boundary) / (boundary - g_ratio))


@pytest.mark.parametrize("dfn", [1, 2, 3])
@pytest.mark.parametrize("n_clusters", [6, 15])
def test_savi_cluster_e_value_has_unit_mean_under_exact_null(dfn, n_clusters):
    """Check the exact e-value property of the clustered formula.

    Under the common-design Gaussian benchmark of Lindon (2026,
    arXiv:2610.25613, Proposition common), (G - q) / (q (G - 1)) Q is
    F(q, G - q), and the Bayes factor has null mean one. The reference is the
    F density alone; an exponent of (G - 1 + q) / 2 fails for q > 1.
    """
    scale = dfn * (n_clusters - 1) / (n_clusters - dfn)
    mean, _ = integrate.quad(
        lambda f: (
            stats.f.pdf(f, dfn, n_clusters - dfn)
            * _savi_cluster_e_value(
                scale * f, dfn=dfn, n_clusters=n_clusters, mixture_precision=1.5
            )
        ),
        0,
        np.inf,
        limit=200,
    )
    np.testing.assert_allclose(mean, 1.0, rtol=1e-6)


@pytest.mark.parametrize(
    "small_sample_correction",
    [ssc(), ssc(k_adj=False), ssc(G_adj=False)],
)
def test_savi_crv1_matches_per_cluster_hotelling_under_common_design(
    small_sample_correction,
):
    """Compare clustered SAVI with Hotelling inference on per-cluster fits.

    With a common within-cluster design, the CR1 t statistic equals the
    one-sample t statistic of the per-cluster OLS coefficients (Lindon 2026,
    arXiv:2610.25613, Proposition common). The reference fits each cluster
    separately and never touches the pooled covariance or `ssc`.
    """
    rng = np.random.default_rng(20261026)
    n_clusters, cluster_size = 12, 5
    x_within = rng.normal(size=cluster_size)
    cluster = np.repeat(np.arange(n_clusters), cluster_size)
    x = np.tile(x_within, n_clusters)
    y = (
        0.4
        + 0.3 * x
        + rng.normal(size=n_clusters)[cluster]
        + rng.normal(size=cluster.size)
    )
    data = pd.DataFrame({"Y": y, "X1": x, "cluster": cluster})
    fit = feols("Y ~ X1", data, vcov={"CRV1": "cluster"}, ssc=small_sample_correction)

    design = np.column_stack([np.ones(cluster_size), x_within])
    per_cluster = np.array(
        [
            np.linalg.lstsq(design, y[cluster == g], rcond=None)[0]
            for g in range(n_clusters)
        ]
    )
    estimate = per_cluster.mean(axis=0)
    se = per_cluster.std(axis=0, ddof=1) / np.sqrt(n_clusters)
    mixture_precision, alpha = 2.0, 0.1

    np.testing.assert_allclose(
        fit.evalue(mixture_precision=mixture_precision),
        _hotelling_e_value(
            (estimate / se) ** 2, n_clusters, dfn=1, mixture_precision=2.0
        ),
        rtol=1e-8,
    )
    radius = _hotelling_radius(alpha, mixture_precision, n_clusters)
    np.testing.assert_allclose(
        fit.confint(
            alpha=alpha, inference_type="savi", mixture_precision=mixture_precision
        ).to_numpy(),
        np.column_stack([estimate - radius * se, estimate + radius * se]),
        rtol=1e-8,
    )


def test_savi_crv1_matches_explicit_cr1_sandwich_with_unequal_clusters():
    """Compare clustered SAVI with an explicit CR1 sandwich.

    Clusters differ in size and design. The reference builds the CR1
    covariance G / (G - 1) (X'X)^-1 sum_g s_g s_g' (X'X)^-1 of Lindon (2026,
    arXiv:2610.25613, equation CR1) from the raw arrays, bypassing pyfixest's
    covariance and its small-sample factor.
    """
    rng = np.random.default_rng(7)
    sizes = rng.integers(1, 9, size=25)
    cluster = np.repeat(np.arange(sizes.size), sizes)
    x = rng.normal(size=cluster.size) + rng.normal(size=sizes.size)[cluster]
    y = (
        1.0
        + 0.2 * x
        + rng.normal(size=sizes.size)[cluster]
        + rng.normal(scale=1 + np.abs(x))
    )
    data = pd.DataFrame({"Y": y, "X1": x, "cluster": cluster})
    fit = feols("Y ~ X1", data, vcov={"CRV1": "cluster"})

    design = np.column_stack([np.ones(cluster.size), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    scores = design * (y - design @ beta)[:, None]
    cluster_scores = np.array(
        [scores[cluster == g].sum(axis=0) for g in range(sizes.size)]
    )
    n_clusters = sizes.size
    vcov = (
        n_clusters
        / (n_clusters - 1)
        * bread
        @ cluster_scores.T
        @ cluster_scores
        @ bread
    )
    se = np.sqrt(np.diag(vcov))
    radius = _hotelling_radius(0.05, 1.0, n_clusters)

    np.testing.assert_allclose(
        fit.evalue(),
        _hotelling_e_value((beta / se) ** 2, n_clusters, dfn=1, mixture_precision=1.0),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        fit.confint(inference_type="savi").to_numpy(),
        np.column_stack([beta - radius * se, beta + radius * se]),
        rtol=1e-8,
    )
