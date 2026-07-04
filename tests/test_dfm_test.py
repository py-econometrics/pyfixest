"""
Tests for the DFM heterogeneity test (Ding, Feller, Miratrix 2019).

Covers the standalone function and the Feols.dfm_test() method.
"""

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation.post_estimation.dfm_test import dfm_test


# ---------------------------------------------------------------------------
# Standalone function tests
# ---------------------------------------------------------------------------


class TestDfmTestStandalone:
    """Tests for the standalone dfm_test function."""

    def test_no_heterogeneity_high_pvalue(self):
        """Under constant treatment effect, the test should not reject."""
        rng = np.random.default_rng(123)
        n = 2000
        X = rng.standard_normal((n, 2))
        D = rng.binomial(1, 0.5, n)
        # constant effect = 1.0, no interaction with X
        Y = 1.0 + X @ np.array([0.5, -0.3]) + D * 1.0 + rng.standard_normal(n)

        result = dfm_test(y=Y, treatment=D, X=X)
        # p-value should be large (fail to reject null)
        assert result["pvalue"] > 0.05, (
            f"Expected high p-value under null, got {result['pvalue']:.4f}"
        )
        assert result["df"] == 2

    def test_strong_heterogeneity_low_pvalue(self):
        """Under strong heterogeneity, the test should reject."""
        rng = np.random.default_rng(456)
        n = 2000
        X = rng.standard_normal((n, 2))
        D = rng.binomial(1, 0.5, n)
        # effect varies strongly with X[:,0]: tau(x) = 1 + 3*x1
        tau = 1.0 + 3.0 * X[:, 0]
        Y = 1.0 + X @ np.array([0.5, -0.3]) + D * tau + rng.standard_normal(n)

        result = dfm_test(y=Y, treatment=D, X=X)
        # p-value should be very small
        assert result["pvalue"] < 0.001, (
            f"Expected low p-value under heterogeneity, got {result['pvalue']:.4f}"
        )
        assert result["df"] == 2

    def test_single_covariate(self):
        """Works with a single covariate (df=1)."""
        rng = np.random.default_rng(789)
        n = 1000
        X = rng.standard_normal((n, 1))
        D = rng.binomial(1, 0.5, n)
        # heterogeneous effect: tau(x) = 2 + 4*x
        Y = 0.5 * X.ravel() + D * (2.0 + 4.0 * X.ravel()) + rng.standard_normal(n)

        result = dfm_test(y=Y, treatment=D, X=X)
        assert result["df"] == 1
        assert result["pvalue"] < 0.001

    def test_matches_manual_computation(self):
        """Verify the statistic matches a hand-rolled computation."""
        rng = np.random.default_rng(42)
        n = 500
        X = rng.standard_normal((n, 1))
        D = rng.binomial(1, 0.5, n)
        Y = X.ravel() + D * (1.0 + 2.0 * X.ravel()) + 0.5 * rng.standard_normal(n)

        # Manual computation
        ones = np.ones((n, 1))
        X_full = np.hstack([ones, X])
        mask1 = D == 1
        mask0 = D == 0
        X1 = X_full[mask1]
        X0 = X_full[mask0]
        y1 = Y[mask1]
        y0 = Y[mask0]
        n1, n0 = X1.shape[0], X0.shape[0]

        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        beta0 = np.linalg.lstsq(X0, y0, rcond=None)[0]
        resid1 = y1 - X1 @ beta1
        resid0 = y0 - X0 @ beta0
        E1 = resid1[:, None] * X1
        E0 = resid0[:, None] * X0

        Sxx1_inv = np.linalg.inv(X1.T @ X1 / n1)
        Sxx0_inv = np.linalg.inv(X0.T @ X0 / n0)
        cov_E1 = np.cov(E1, rowvar=False)
        cov_E0 = np.cov(E0, rowvar=False)
        cov_beta = (
            Sxx1_inv @ (cov_E1 / n1) @ Sxx1_inv
            + Sxx0_inv @ (cov_E0 / n0) @ Sxx0_inv
        )

        beta_diff = beta1 - beta0
        beta1_hat = beta_diff[1:]
        cov_beta1 = cov_beta[1:, 1:]
        expected_stat = float(beta1_hat @ np.linalg.solve(cov_beta1, beta1_hat))

        result = dfm_test(y=Y, treatment=D, X=X)
        np.testing.assert_allclose(result["statistic"], expected_stat, rtol=1e-10)

    def test_unbalanced_treatment(self):
        """Works with unbalanced treatment assignment (e.g. 80/20 split)."""
        rng = np.random.default_rng(101)
        n = 1000
        X = rng.standard_normal((n, 2))
        D = rng.binomial(1, 0.2, n)  # only 20% treated
        tau = 1.0 + 2.5 * X[:, 1]
        Y = X @ np.array([1.0, 0.5]) + D * tau + rng.standard_normal(n)

        result = dfm_test(y=Y, treatment=D, X=X)
        # should still detect heterogeneity
        assert result["pvalue"] < 0.01
        assert result["df"] == 2

    def test_1d_covariate_input(self):
        """Accepts a 1-D array as X (single covariate)."""
        rng = np.random.default_rng(202)
        n = 500
        X = rng.standard_normal(n)  # 1-D
        D = rng.binomial(1, 0.5, n)
        Y = X + D * (1.0 + 3.0 * X) + rng.standard_normal(n)

        result = dfm_test(y=Y, treatment=D, X=X)
        assert result["df"] == 1
        assert result["pvalue"] < 0.01

    # --- Error cases ---

    def test_non_binary_treatment_raises(self):
        """Non-binary treatment raises ValueError."""
        X = np.random.randn(100, 2)
        D = np.random.choice([0, 1, 2], 100)
        Y = np.random.randn(100)
        with pytest.raises(ValueError, match="binary"):
            dfm_test(y=Y, treatment=D, X=X)

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise ValueError."""
        X = np.random.randn(100, 2)
        D = np.random.binomial(1, 0.5, 50)
        Y = np.random.randn(100)
        with pytest.raises(ValueError, match="same length"):
            dfm_test(y=Y, treatment=D, X=X)

    def test_too_few_obs_raises(self):
        """Too few observations in one arm raises ValueError."""
        # 5 obs total, 3 covariates + intercept = need K=4 per arm
        X = np.random.randn(5, 3)
        D = np.array([1, 1, 0, 0, 0])
        Y = np.random.randn(5)
        with pytest.raises(ValueError, match="Not enough observations"):
            dfm_test(y=Y, treatment=D, X=X)


# ---------------------------------------------------------------------------
# Feols method tests (integration)
# ---------------------------------------------------------------------------


class TestDfmTestFeols:
    """Tests for the Feols.dfm_test() method."""

    @pytest.fixture
    def heterogeneous_data(self):
        """Generate data with known heterogeneous treatment effects."""
        rng = np.random.default_rng(42)
        n = 1000
        X1 = rng.standard_normal(n)
        X2 = rng.standard_normal(n)
        D = rng.binomial(1, 0.5, n)
        # tau(x) = 1 + 2*X1 (heterogeneous in X1, not X2)
        Y = 1.0 + 0.5 * X1 - 0.3 * X2 + D * (1.0 + 2.0 * X1) + rng.standard_normal(n)
        return pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})

    @pytest.fixture
    def homogeneous_data(self):
        """Generate data with constant treatment effect."""
        rng = np.random.default_rng(99)
        n = 2000
        X1 = rng.standard_normal(n)
        X2 = rng.standard_normal(n)
        D = rng.binomial(1, 0.5, n)
        Y = 1.0 + 0.5 * X1 - 0.3 * X2 + D * 2.0 + rng.standard_normal(n)
        return pd.DataFrame({"Y": Y, "D": D, "X1": X1, "X2": X2})

    def test_detects_heterogeneity(self, heterogeneous_data):
        """Method detects heterogeneity and returns low p-value."""
        import pyfixest as pf

        fit = pf.feols("Y ~ D + X1 + X2", data=heterogeneous_data)
        res = fit.dfm_test(treatment="D")

        assert isinstance(res, pd.Series)
        assert "statistic" in res.index
        assert "pvalue" in res.index
        assert "df" in res.index
        assert res["pvalue"] < 0.001
        assert res["df"] == 2

    def test_no_false_positive(self, homogeneous_data):
        """Method does not reject under constant effect."""
        import pyfixest as pf

        fit = pf.feols("Y ~ D + X1 + X2", data=homogeneous_data)
        res = fit.dfm_test(treatment="D")

        assert res["pvalue"] > 0.05

    def test_stores_attributes(self, heterogeneous_data):
        """Method stores detailed results on the model object."""
        import pyfixest as pf

        fit = pf.feols("Y ~ D + X1 + X2", data=heterogeneous_data)
        fit.dfm_test(treatment="D")

        assert hasattr(fit, "_dfm_statistic")
        assert hasattr(fit, "_dfm_pvalue")
        assert hasattr(fit, "_dfm_df")
        assert hasattr(fit, "_dfm_beta_hat")
        assert hasattr(fit, "_dfm_cov_beta")
        # beta_hat should have 3 elements (intercept + 2 covariates)
        assert len(fit._dfm_beta_hat) == 3

    def test_fixed_effects_raises(self, heterogeneous_data):
        """Raises NotImplementedError for models with fixed effects."""
        import pyfixest as pf

        heterogeneous_data["group"] = np.random.choice(
            range(10), size=len(heterogeneous_data)
        )
        fit = pf.feols("Y ~ D + X1 | group", data=heterogeneous_data)
        with pytest.raises(NotImplementedError, match="fixed effects"):
            fit.dfm_test(treatment="D")

    def test_missing_treatment_raises(self, heterogeneous_data):
        """Raises ValueError if treatment variable not in model."""
        import pyfixest as pf

        fit = pf.feols("Y ~ X1 + X2", data=heterogeneous_data)
        with pytest.raises(ValueError, match="not found"):
            fit.dfm_test(treatment="D")

    def test_non_binary_treatment_raises(self):
        """Raises ValueError if treatment is not 0/1."""
        import pyfixest as pf

        rng = np.random.default_rng(55)
        n = 200
        data = pd.DataFrame({
            "Y": rng.standard_normal(n),
            "D": rng.choice([0, 1, 2], n),
            "X1": rng.standard_normal(n),
        })
        fit = pf.feols("Y ~ D + X1", data=data)
        with pytest.raises(ValueError, match="binary"):
            fit.dfm_test(treatment="D")

    def test_no_covariates_raises(self):
        """Raises ValueError if model has no covariates besides treatment."""
        import pyfixest as pf

        rng = np.random.default_rng(77)
        n = 200
        data = pd.DataFrame({
            "Y": rng.standard_normal(n),
            "D": rng.binomial(1, 0.5, n),
        })
        fit = pf.feols("Y ~ D", data=data)
        with pytest.raises(ValueError, match="at least one covariate"):
            fit.dfm_test(treatment="D")

    def test_consistent_with_standalone(self, heterogeneous_data):
        """Feols method gives same result as calling standalone directly."""
        import pyfixest as pf

        fit = pf.feols("Y ~ D + X1 + X2", data=heterogeneous_data)
        res_method = fit.dfm_test(treatment="D")

        # Call standalone directly with same data
        y = heterogeneous_data["Y"].to_numpy()
        D = heterogeneous_data["D"].to_numpy()
        X = heterogeneous_data[["X1", "X2"]].to_numpy()
        res_standalone = dfm_test(y=y, treatment=D, X=X)

        np.testing.assert_allclose(
            res_method["statistic"], res_standalone["statistic"], rtol=1e-10
        )
        np.testing.assert_allclose(
            res_method["pvalue"], res_standalone["pvalue"], rtol=1e-10
        )
