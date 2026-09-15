"""
Validate the Python DFM heterogeneity test against the reference R.

Compares `pyfixest..._dfm_heterogeneity_test` to the vendored, MIT-licensed reference
implementation `dfmTest` from Netflix-Skunkworks/causaltransportr, committed
verbatim under `tests/vendored/causaltransportr/` (see the LICENSE there). The R function is
pure base R, so this runs wherever the R test environment is available.
"""

import os

import numpy as np
import pytest
import rpy2.robjects as ro

from pyfixest.estimation.post_estimation.dfm_test import _dfm_heterogeneity_test

# Both sides are closed-form OLS + a sandwich; agreement is to machine precision
# up to BLAS reduction order.
rtol = 1e-8

_R_FILE = os.path.join(
    os.path.dirname(__file__), "vendored", "causaltransportr", "dfmTest.R"
)
ro.r["source"](_R_FILE)
_r_dfmTest = ro.globalenv["dfmTest"]


def _run_r(y, a, X):
    """Call the vendored R dfmTest and return [statistic, pvalue].

    Build R atomic vectors and a matrix explicitly: numpy2ri gives a 1-D array a
    `dim` attribute, which propagates to `lm.fit`'s residuals and breaks the
    `residuals * X` product inside dfmTest.
    """
    y = np.asarray(y, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    r_X = ro.r["matrix"](ro.FloatVector(X.flatten(order="F")), nrow=n, ncol=p)
    out = _r_dfmTest(ro.FloatVector(y), ro.FloatVector(a), r_X)
    return np.asarray(out).ravel()


def _make_data(seed, n, p, p_treat, hetero):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    D = rng.binomial(1, p_treat, n)
    # under hetero the effect is modified by the first covariate; otherwise constant
    tau = 1.0 + (2.0 * X[:, 0] if hetero else 0.0)
    Y = X @ rng.standard_normal(p) + D * tau + rng.standard_normal(n)
    return Y, D, X


@pytest.mark.against_r_core
@pytest.mark.parametrize(
    "seed, n, p, p_treat, hetero",
    [
        (1, 1000, 1, 0.5, True),  # single covariate, strong heterogeneity
        (2, 1000, 3, 0.5, True),  # several covariates, balanced arms
        (3, 1500, 2, 0.2, True),  # unbalanced 80/20 treatment
        (4, 800, 4, 0.65, False),  # null holds -> nonzero p-value to compare
        (5, 1200, 2, 0.5, False),  # null holds, balanced
    ],
)
def test_dfm_matches_r(seed, n, p, p_treat, hetero):
    """Python statistic and p-value match the reference R implementation."""
    Y, D, X = _make_data(seed, n, p, p_treat, hetero)

    py = _dfm_heterogeneity_test(y=Y, treatment=D, X=X)
    r_stat, r_pval = _run_r(Y, D, X)

    np.testing.assert_allclose(py["statistic"], r_stat, rtol=rtol)
    np.testing.assert_allclose(py["pvalue"], r_pval, rtol=rtol)
    assert py["df"] == p
