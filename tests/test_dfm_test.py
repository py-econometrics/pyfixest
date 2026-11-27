"""
Tests for the DFM heterogeneity test (Ding, Feller, Miratrix 2019).

Numerical validation against the reference R implementation lives in
`test_dfm_test_vs_r.py`. This file covers the Python-only paths: the Feols
method reproduces the standalone function, regressors are read from the model
matrix (so transforms work), and -- last -- the power/size properties.

Error and unsupported-model paths live in `test_errors.py`.
"""

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.post_estimation.dfm_test import (
    EffectHeterogeneityTest,
    _dfm_heterogeneity_test,
)


@pytest.fixture
def heterogeneous_data():
    """Return data whose treatment effect varies with X1 (tau = 1 + 2*X1)."""
    rng = np.random.default_rng(42)
    n = 1000
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    D = rng.binomial(1, 0.5, n)
    Y = 1.0 + 0.5 * X1 - 0.3 * X2 + D * (1.0 + 2.0 * X1) + rng.standard_normal(n)
    return pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})


@pytest.fixture
def homogeneous_data():
    """Return data with a constant treatment effect (no heterogeneity)."""
    rng = np.random.default_rng(99)
    n = 2000
    X1 = rng.standard_normal(n)
    X2 = rng.standard_normal(n)
    D = rng.binomial(1, 0.5, n)
    Y = 1.0 + 0.5 * X1 - 0.3 * X2 + D * 2.0 + rng.standard_normal(n)
    return pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})


def test_method_matches_standalone(heterogeneous_data):
    """Feols.dfm_heterogeneity_test reproduces the standalone function on the same design."""
    fit = pf.feols("Y ~ D + X1 + X2", data=heterogeneous_data)
    res = fit.dfm_heterogeneity_test(treatment="D")

    assert isinstance(res, EffectHeterogeneityTest)

    y = heterogeneous_data["Y"].to_numpy()
    treated = heterogeneous_data["D"].to_numpy() == 1
    X = heterogeneous_data[["X1", "X2"]].to_numpy()
    ref = _dfm_heterogeneity_test(y=y, treated=treated, X=X)

    np.testing.assert_allclose(res.stat, ref.stat, rtol=1e-10)
    np.testing.assert_allclose(res.pvalue, ref.pvalue, rtol=1e-10)
    assert res.df == 2


def test_regressors_read_from_model_matrix():
    """A transformed regressor is read from the model matrix, not the raw data."""
    rng = np.random.default_rng(21)
    n = 1000
    X1 = rng.uniform(1.0, 5.0, n)  # positive so np.log is defined
    X2 = rng.standard_normal(n)
    D = rng.binomial(1, 0.5, n)
    Y = np.log(X1) + D * (1.0 + 2.0 * np.log(X1)) + rng.standard_normal(n)
    data = pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})

    fit = pf.feols("Y ~ D + np.log(X1) + X2", data=data)
    res = fit.dfm_heterogeneity_test(treatment="D")

    ref = _dfm_heterogeneity_test(
        y=Y, treated=D == 1, X=np.column_stack([np.log(X1), X2])
    )
    assert res.df == 2
    np.testing.assert_allclose(res.stat, ref.stat, rtol=1e-10)


# --- power and size (property-based, come last) -----------------------------


def test_detects_heterogeneity(heterogeneous_data):
    """Under strong heterogeneity, the test rejects the null."""
    fit = pf.feols("Y ~ D + X1 + X2", data=heterogeneous_data)
    assert fit.dfm_heterogeneity_test(treatment="D").pvalue < 0.001


def test_no_false_positive(homogeneous_data):
    """Under a constant treatment effect, the test does not reject."""
    fit = pf.feols("Y ~ D + X1 + X2", data=homogeneous_data)
    assert fit.dfm_heterogeneity_test(treatment="D").pvalue > 0.05
