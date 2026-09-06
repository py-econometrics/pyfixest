"""
Compare the GLM cluster jackknife (`CRV3`) against leave-one-cluster-out refits in R.

The reference is base R's `stats::glm()`: each leave-one-cluster-out fit is
estimated in R, and the resulting coefficients are aggregated with the cluster
jackknife of Mackinnon, Nielsen & Webb (2022), which centers at the full-sample
estimate. Small-sample corrections are switched off on both sides so that the
comparison is about the jackknife itself.
"""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf

stats = importr("stats")

# IRLS in pyfixest and R's glm converge to the same coefficients but stop on
# different criteria, so allow a small relative difference in the refits.
rtol = 1e-06
atol = 1e-08

R_FAMILY = {
    "logit": 'binomial(link = "logit")',
    "probit": 'binomial(link = "probit")',
    "poisson": 'poisson(link = "log")',
    "gaussian": 'gaussian(link = "identity")',
}


@pytest.fixture
def glm_cluster_data():
    "Return clustered data with a binary, a count, and a continuous response."
    rng = np.random.default_rng(20260906)
    N = 300
    G = 10
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=N),
            "x2": rng.normal(size=N),
            "cluster": np.arange(N) % G,
        }
    )
    eta = 0.5 * data["x1"] - 0.3 * data["x2"]
    data["y_bin"] = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    data["y_count"] = rng.poisson(np.exp(eta))
    data["y_cont"] = eta + rng.normal(size=N)
    return data


def _r_cluster_jackknife(data: pd.DataFrame, fml: str, family: str) -> pd.DataFrame:
    "Fit `fml` in R once per left-out cluster and return the coefficients."
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv["df"] = data
    ro.r(f"""
        fit_all <- glm({fml}, family = {R_FAMILY[family]}, data = df)
        clusters <- sort(unique(df$cluster))
        beta_jack <- t(sapply(clusters, function(g) {{
            coef(glm({fml}, family = {R_FAMILY[family]}, data = df[df$cluster != g, ]))
        }}))
        coef_names <- names(coef(fit_all))
        beta_full <- coef(fit_all)
    """)
    with (ro.default_converter + pandas2ri.converter).context():
        beta_jack = np.asarray(ro.globalenv["beta_jack"])
        beta_full = np.asarray(ro.globalenv["beta_full"])
        coef_names = list(ro.globalenv["coef_names"])

    return pd.DataFrame(beta_jack, columns=coef_names), pd.Series(
        beta_full, index=coef_names
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "family, depvar",
    [
        ("logit", "y_bin"),
        ("probit", "y_bin"),
        ("poisson", "y_count"),
        ("gaussian", "y_cont"),
    ],
)
def test_glm_crv3_matches_r_cluster_jackknife(glm_cluster_data, family, depvar):
    "The GLM jackknife equals the same aggregation over R's leave-one-out fits."
    fml = f"{depvar} ~ x1 + x2"
    fit = pf.feglm(
        fml,
        data=glm_cluster_data,
        family=family,
        vcov={"CRV3": "cluster"},
        ssc=pf.ssc(k_adj=False, G_adj=False),
    )

    beta_jack, beta_full = _r_cluster_jackknife(glm_cluster_data, fml, family)
    beta_jack = beta_jack.rename(columns={"(Intercept)": "Intercept"})
    beta_full = beta_full.rename({"(Intercept)": "Intercept"})

    # coefficients first: a jackknife of the wrong estimator would still be
    # positive definite, so compare the refit inputs, not only their spread.
    np.testing.assert_allclose(
        fit.coef().to_numpy(),
        beta_full[fit._coefnames].to_numpy(),
        rtol=rtol,
        atol=atol,
    )

    centered = beta_jack[fit._coefnames].to_numpy() - beta_full[fit._coefnames].to_numpy()
    expected_vcov = centered.T @ centered

    np.testing.assert_allclose(fit._vcov, expected_vcov, rtol=rtol, atol=atol)


@pytest.mark.against_r_core
def test_glm_crv3_small_sample_correction(glm_cluster_data):
    "The default correction rescales the jackknife by G / (G - 1) and N / (N - k)."
    fml = "y_bin ~ x1 + x2"
    unadjusted = pf.feglm(
        fml,
        data=glm_cluster_data,
        family="logit",
        vcov={"CRV3": "cluster"},
        ssc=pf.ssc(k_adj=False, G_adj=False),
    )
    adjusted = pf.feglm(
        fml, data=glm_cluster_data, family="logit", vcov={"CRV3": "cluster"}
    )

    G = glm_cluster_data["cluster"].nunique()
    N = len(glm_cluster_data)
    k = len(adjusted._coefnames)
    factor = (G / (G - 1)) * ((N - 1) / (N - k))

    np.testing.assert_allclose(
        adjusted._vcov, factor * unadjusted._vcov, rtol=1e-10, atol=1e-12
    )
