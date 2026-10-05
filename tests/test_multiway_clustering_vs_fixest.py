"""Live R fixest coverage for multiway CRV1 covariance and SSC options.

Keep the complete correction matrix separate from the general estimator tests.
The seeded fixture avoids eigenvalue repair so comparisons exercise the raw
inclusion-exclusion estimator; repair remains a documented compatibility gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.utils.utils import ssc

fixest = importr("fixest")
stats = importr("stats")


@pytest.fixture(scope="module")
def multiway_cluster_data():
    """Seeded outcomes and crossed clusters shared by the live-R comparisons."""
    rng = np.random.default_rng(20260922)
    n = 1200
    data = pd.DataFrame(
        {f"c{i}": rng.integers(g, size=n) for i, g in enumerate([12, 17, 23, 29], 1)}
    )
    data["fe"] = rng.integers(8, size=n)
    data["x"] = rng.normal(size=n)
    data["z"] = rng.normal(size=n)
    data["d"] = data.z + rng.normal(size=n)
    shock = sum(
        rng.normal(scale=0.3, size=data[c].max() + 1)[data[c]]
        for c in ["c1", "c2", "c3", "c4"]
    )
    eta = 0.2 + 0.3 * data.x + shock
    data["y"] = eta + 0.4 * data.d + rng.normal(size=n)
    data["count"] = rng.poisson(np.exp(eta))
    data["binary"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    data["w"] = rng.integers(1, 4, size=n)
    return data


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "fml",
    [
        pytest.param("y ~ x", id="no-fe"),
        pytest.param("y ~ x | c1", id="nested-fe"),
        pytest.param("y ~ x | fe", id="nonnested-fe"),
        pytest.param("y ~ x | c1 + fe", id="mixed-fe"),
    ],
)
@pytest.mark.parametrize("n_clusters", [1, 2, 3])
@pytest.mark.parametrize("weights_type", [None, "aweights"])
@pytest.mark.parametrize("k_adj", [False, True])
@pytest.mark.parametrize("G_adj", [False, True])
@pytest.mark.parametrize("k_fixef", ["none", "full", "nonnested"])
@pytest.mark.parametrize("G_df", ["min", "conventional"])
def test_multiway_clustering_against_fixest(
    multiway_cluster_data, fml, n_clusters, weights_type, k_adj, G_adj, k_fixef, G_df
):
    """Cross OLS FE/nesting cases, cluster counts, weights, and SSC options.

    Nesting is determined by the formula: c1 is always a cluster dimension,
    whereas fe is independently generated. Keep the Cartesian matrix explicit,
    including equivalent SSC settings for one-way clustering and no-FE fits.
    """
    _assert_multiway_clustering_against_fixest(
        data=multiway_cluster_data,
        model="feols",
        fml=fml,
        n_clusters=n_clusters,
        weights_type=weights_type,
        k_adj=k_adj,
        G_adj=G_adj,
        k_fixef=k_fixef,
        G_df=G_df,
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("G_df", ["min", "conventional"])
@pytest.mark.parametrize(
    "model,fml,r_fml,weights_type,n_clusters,ambiguous_labels",
    [
        pytest.param(
            "feols",
            "y ~ x + [d ~ z] | c1",
            "y ~ x | c1 | d ~ z",
            "aweights",
            3,
            False,
            id="iv",
        ),
        pytest.param(
            "fepois",
            "count ~ x | c1",
            None,
            "aweights",
            3,
            False,
            id="poisson",
        ),
        pytest.param(
            "logit",
            "binary ~ x | c1",
            None,
            None,
            3,
            False,
            id="binomial-glm",
        ),
        pytest.param(
            "feols",
            "y ~ x | c1 + fe",
            None,
            None,
            4,
            False,
            id="fourway",
        ),
        pytest.param(
            "feols",
            "y ~ x | c1",
            None,
            "fweights",
            3,
            False,
            id="frequency-weights",
        ),
        pytest.param(
            "feols",
            "y ~ x",
            None,
            None,
            3,
            True,
            id="ambiguous-labels",
        ),
    ],
)
def test_multiway_clustering_integration_against_fixest(
    multiway_cluster_data,
    G_df,
    model,
    fml,
    r_fml,
    weights_type,
    n_clusters,
    ambiguous_labels,
):
    """Check estimator-specific inputs and edge cases outside the OLS matrix."""
    data = multiway_cluster_data.copy()
    if ambiguous_labels:
        # ("a-b", "c") and ("a", "b-c") both become "a-b-c" if joined with "-".
        # Their intersections must stay distinct and match R fixest inference.
        data["c1"] = data.c1.map(lambda g: {0: "a-b", 1: "a"}.get(g, f"c1-{g}"))
        data["c2"] = data.c2.map(lambda g: {0: "c", 1: "b-c"}.get(g, f"c2-{g}"))
    _assert_multiway_clustering_against_fixest(
        data=data,
        model=model,
        fml=fml,
        r_fml=r_fml,
        n_clusters=n_clusters,
        weights_type=weights_type,
        k_adj=True,
        G_adj=True,
        k_fixef="nonnested",
        G_df=G_df,
    )


def _assert_multiway_clustering_against_fixest(
    *,
    data,
    model,
    fml,
    n_clusters,
    weights_type,
    k_adj,
    G_adj,
    k_fixef,
    G_df,
    r_fml=None,
):
    """Compare named estimates, covariance, inference, and counts with fixest."""
    cluster = "+".join(f"c{i}" for i in range(1, n_clusters + 1))
    kwargs = (
        {} if weights_type is None else {"weights": "w", "weights_type": weights_type}
    )
    r_kwargs = {}
    r_data = data
    if weights_type == "aweights":
        r_kwargs["weights"] = ro.Formula("~w")
    elif weights_type == "fweights":
        # fixest frequency weights are represented by literal row replication.
        r_data = data.loc[data.index.repeat(data.w)].reset_index(drop=True)
    if model in ("feols", "fepois"):
        py_estimator, r_estimator = getattr(pf, model), getattr(fixest, model)
    else:
        py_estimator, r_estimator = pf.feglm, fixest.feglm
        kwargs["family"] = model
        r_kwargs["family"] = stats.binomial(link=model)
    fit = py_estimator(
        fml=fml,
        data=data,
        vcov={"CRV1": cluster},
        ssc=ssc(k_adj=k_adj, G_adj=G_adj, k_fixef=k_fixef, G_df=G_df),
        **kwargs,
    )
    r_fit = r_estimator(
        ro.Formula(fml if r_fml is None else r_fml),
        data=r_data,
        # fixest 0.14.0 needs an explicit fourway type for four-variable formulas.
        vcov=ro.Formula(("fourway" if n_clusters == 4 else "cluster") + "~" + cluster),
        ssc=fixest.ssc(k_adj, k_fixef, False, G_adj, G_df, "min"),
        **r_kwargs,
    )
    ro.globalenv["multiway_fit"] = r_fit
    r_names = list(ro.r("names(coef(multiway_fit))"))
    r_vcov_names = list(ro.r("rownames(vcov(multiway_fit))"))
    expected_names = [
        {"Intercept": "(Intercept)", "d": "fit_d"}.get(name, name)
        for name in fit.coef().index
    ]
    assert set(expected_names) == set(r_names), "multiway coefficient names"
    assert set(expected_names) == set(r_vcov_names), "multiway covariance names"
    order = [r_names.index(name) for name in expected_names]
    vcov_order = [r_vcov_names.index(name) for name in expected_names]
    # Cluster reductions can differ in accumulation order; IRLS adds stopping
    # error, so allow 1e-6 for GLM inference versus 1e-8 for linear fits.
    inference_atol = 1e-8 if model == "feols" else 1e-6
    np.testing.assert_allclose(
        fit.coef(),
        np.asarray(stats.coef(r_fit))[order],
        rtol=0,
        atol=1e-8,
        err_msg="multiway coefficients",
    )
    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        np.asarray(stats.vcov(r_fit))[np.ix_(vcov_order, vcov_order)],
        rtol=0,
        atol=inference_atol,
        err_msg="multiway covariance",
    )
    np.testing.assert_allclose(
        fit.se(),
        np.asarray(fixest.se(r_fit))[order],
        rtol=0,
        atol=inference_atol,
        err_msg="multiway standard errors",
    )
    np.testing.assert_allclose(
        fit.pvalue(),
        np.asarray(fixest.pvalue(r_fit))[order],
        rtol=0,
        atol=inference_atol,
        err_msg="multiway p-values",
    )
    assert fit.variance_covariance.df_k == int(
        ro.r('attr(multiway_fit$cov.scaled, "df.K")')[0]
    ), "multiway df_k"
    assert fit.variance_covariance.df_t == int(
        ro.r('attr(multiway_fit$cov.scaled, "df.t")')[0]
    ), "multiway df_t"
    assert fit.sample_info.n_obs == int(stats.nobs(r_fit)[0]), "multiway observations"
    assert len(fit.variance_covariance.G) == 2**n_clusters - 1
