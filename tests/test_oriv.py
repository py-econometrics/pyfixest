"""Tests for ORIV (Obviously Related Instrumental Variables) estimation."""

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation.api.oriv import oriv

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simulated_data():
    """
    Simulate data following the training/ability example from Gillen et al.

    DGP:
        ability ~ N(100, 15^2)
        training = 1(ability + noise >= 100)
        sales = 40000 + 5000*training + 100*ability + epsilon
        abilityTest1 = ability + measurement_noise_1
        abilityTest2 = ability + measurement_noise_2
    """
    np.random.seed(12345)
    n = 5000
    df = pd.DataFrame()
    df["ability"] = np.random.normal(100, 15, n)
    df["training"] = (df["ability"] + np.random.normal(0, 10, n) >= 100).astype(int)
    df["sales"] = (
        40000
        + df["training"] * 5000
        + df["ability"] * 100
        + np.random.normal(0, 4000, n)
    )
    df["abilityTest1"] = df["ability"] + np.random.normal(0, 8, n)
    df["abilityTest2"] = df["ability"] + np.random.normal(0, 8, n)
    df["abilityTest3"] = df["ability"] + np.random.normal(0, 8, n)
    return df


@pytest.fixture
def simple_data():
    """Small deterministic dataset for exact value testing."""
    np.random.seed(42)
    n = 200
    x_true = np.random.normal(0, 1, n)
    treatment = np.random.binomial(1, 0.5, n)
    y = 2.0 * treatment + 3.0 * x_true + np.random.normal(0, 0.5, n)
    proxy1 = x_true + np.random.normal(0, 0.3, n)
    proxy2 = x_true + np.random.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "y": y,
            "treatment": treatment,
            "proxy1": proxy1,
            "proxy2": proxy2,
            "x_true": x_true,
        }
    )


# ============================================================================
# Input validation tests
# ============================================================================


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_fewer_than_two_proxies_raises(self, simulated_data):
        with pytest.raises(ValueError, match="at least 2 proxies"):
            oriv("sales ~ training", ["abilityTest1"], simulated_data)

    def test_empty_proxies_raises(self, simulated_data):
        with pytest.raises(ValueError, match="at least 2 proxies"):
            oriv("sales ~ training", [], simulated_data)

    def test_missing_proxy_column_raises(self, simulated_data):
        with pytest.raises(ValueError, match="not found in data"):
            oriv("sales ~ training", ["abilityTest1", "nonexistent"], simulated_data)

    def test_multiple_missing_proxies_raises(self, simulated_data):
        with pytest.raises(ValueError, match="not found in data"):
            oriv("sales ~ training", ["foo", "bar"], simulated_data)


# ============================================================================
# Basic functionality tests
# ============================================================================


class TestBasicFunctionality:
    """Test that ORIV produces correct estimates."""

    def test_oriv_reduces_attenuation_bias(self, simulated_data):
        """ORIV should get closer to true effect than OLS with noisy proxy."""
        from pyfixest.estimation.api.feols import feols

        # OLS with noisy proxy (attenuated)
        ols_noisy = feols("sales ~ training + abilityTest1", data=simulated_data)
        training_ols = ols_noisy.coef()["training"]

        # ORIV (corrected)
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        training_oriv = result.coef()["training"]

        # True effect is 5000
        # OLS with noisy proxy overestimates (due to attenuation of ability coef)
        # ORIV should be closer to 5000
        assert abs(training_oriv - 5000) < abs(training_ols - 5000)

    def test_oriv_ability_coefficient_close_to_true(self, simulated_data):
        """The instrumented variable coefficient should be close to 100."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        ability_coef = result.coef()["ability_proxy"]
        # True coefficient is 100, allow some tolerance
        assert abs(ability_coef - 100) < 10

    def test_oriv_training_coefficient_close_to_true(self, simulated_data):
        """The training coefficient should be close to 5000."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        training_coef = result.coef()["training"]
        # True coefficient is 5000, allow some tolerance
        assert abs(training_coef - 5000) < 500

    def test_oriv_returns_feols_object(self, simulated_data):
        """The result should be a Feols (or Feiv) object with standard methods."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        # Should have standard methods
        assert hasattr(result, "coef")
        assert hasattr(result, "se")
        assert hasattr(result, "tstat")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "confint")
        assert hasattr(result, "summary")


# ============================================================================
# Multiple proxies tests
# ============================================================================


class TestMultipleProxies:
    """Test ORIV with more than 2 proxies."""

    def test_three_proxies(self, simulated_data):
        """ORIV should work with 3 proxies."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2", "abilityTest3"],
            simulated_data,
            "ability_proxy",
        )
        training_coef = result.coef()["training"]
        ability_coef = result.coef()["ability_proxy"]

        # Should still be close to true values
        assert abs(training_coef - 5000) < 500
        assert abs(ability_coef - 100) < 10

    def test_three_proxies_more_efficient(self, simulated_data):
        """Using 3 proxies should give smaller SE than 2 proxies."""
        result_2 = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        result_3 = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2", "abilityTest3"],
            simulated_data,
            "ability_proxy",
        )
        # More instruments should reduce SE of the instrumented variable
        se_2 = result_2.se()["ability_proxy"]
        se_3 = result_3.se()["ability_proxy"]
        assert se_3 < se_2


# ============================================================================
# Variance-covariance tests
# ============================================================================


class TestVcov:
    """Test different vcov specifications."""

    def test_default_vcov_is_crv1(self, simulated_data):
        """Default should use CRV1 clustered on observation ID."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        # CRV1 SEs should be larger than iid SEs due to within-obs correlation
        result_iid = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
            vcov="iid",
        )
        # Clustered SEs should generally be larger
        assert result.se()["ability_proxy"] > result_iid.se()["ability_proxy"]

    def test_custom_vcov_iid(self, simulated_data):
        """Should accept iid vcov."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
            vcov="iid",
        )
        assert result.coef()["training"] is not None

    def test_custom_vcov_hetero(self, simulated_data):
        """Should accept HC1 vcov."""
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
            vcov="hetero",
        )
        assert result.coef()["training"] is not None


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_input_dataframe_not_modified(self, simulated_data):
        """The input DataFrame should not be modified."""
        original_cols = set(simulated_data.columns)
        original_shape = simulated_data.shape
        oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        assert set(simulated_data.columns) == original_cols
        assert simulated_data.shape == original_shape

    def test_var_name_does_not_collide_with_existing_columns(self, simulated_data):
        """Using a var_name that exists in the data should still work."""
        # 'ability' is already a column, but the stacked data uses it as the
        # instrumented variable name
        result = oriv(
            "sales ~ training",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability",
        )
        assert "ability" in result.coef().index

    def test_formula_with_multiple_exogenous(self, simulated_data):
        """Should work with multiple exogenous regressors."""
        simulated_data = simulated_data.copy()
        simulated_data["x2"] = np.random.normal(0, 1, len(simulated_data))
        result = oriv(
            "sales ~ training + x2",
            ["abilityTest1", "abilityTest2"],
            simulated_data,
            "ability_proxy",
        )
        assert "training" in result.coef().index
        assert "x2" in result.coef().index
        assert "ability_proxy" in result.coef().index

    def test_small_sample(self):
        """Should work with a small sample."""
        np.random.seed(99)
        n = 30
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 1, n)
        p1 = x + np.random.normal(0, 0.5, n)
        p2 = x + np.random.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "p1": p1, "p2": p2})
        result = oriv("y ~ 1", ["p1", "p2"], df, "x_hat", vcov="iid")
        # Should run without error and produce a coefficient
        assert "x_hat" in result.coef().index

    def test_proxies_with_different_noise_levels(self):
        """Should work even when proxies have very different noise levels."""
        np.random.seed(123)
        n = 1000
        x = np.random.normal(0, 1, n)
        y = 3.0 * x + np.random.normal(0, 0.5, n)
        # One very noisy proxy, one less noisy
        p1 = x + np.random.normal(0, 0.1, n)  # low noise
        p2 = x + np.random.normal(0, 2.0, n)  # high noise
        df = pd.DataFrame({"y": y, "p1": p1, "p2": p2})
        result = oriv("y ~ 1", ["p1", "p2"], df, "x_hat", vcov="iid")
        x_coef = result.coef()["x_hat"]
        # Should still be close to 3.0
        assert abs(x_coef - 3.0) < 1.0


# ============================================================================
# Consistency tests
# ============================================================================


class TestConsistency:
    """Test that results are consistent with known properties."""

    def test_oriv_equals_standard_iv_for_two_proxies(self, simple_data):
        """
        With 2 proxies, ORIV is equivalent to running 2SLS on the stacked data
        where each proxy instruments the other.
        """
        from pyfixest.estimation.api.feols import feols

        # Run ORIV
        result_oriv = oriv("y ~ treatment", ["proxy1", "proxy2"], simple_data, "x_hat")

        # Manually construct the stacked data and run feols IV
        n = len(simple_data)
        df1 = simple_data.copy()
        df1["_obs_id"] = np.arange(n)
        df1["_copy"] = 0
        df1["x_hat"] = df1["proxy1"]
        df1["_iv"] = df1["proxy2"]

        df2 = simple_data.copy()
        df2["_obs_id"] = np.arange(n)
        df2["_copy"] = 1
        df2["x_hat"] = df2["proxy2"]
        df2["_iv"] = df2["proxy1"]

        df_stacked = pd.concat([df1, df2], ignore_index=True)
        df_stacked["_d1"] = (df_stacked["_copy"] == 1).astype(float)

        result_manual = feols(
            "y ~ treatment + _d1 | x_hat ~ _iv",
            data=df_stacked,
            vcov={"CRV1": "_obs_id"},
        )

        # Coefficients should be identical
        np.testing.assert_allclose(
            result_oriv.coef()["treatment"],
            result_manual.coef()["treatment"],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result_oriv.coef()["x_hat"],
            result_manual.coef()["x_hat"],
            rtol=1e-10,
        )

    def test_perfect_proxies_recover_ols(self):
        """
        If proxies have zero measurement error, ORIV should give the same
        result as OLS with the true variable.
        """
        from pyfixest.estimation.api.feols import feols

        np.random.seed(77)
        n = 500
        x = np.random.normal(0, 1, n)
        t = np.random.binomial(1, 0.5, n)
        y = 2.0 * t + 3.0 * x + np.random.normal(0, 0.5, n)
        # Perfect proxies (no noise)
        df = pd.DataFrame({"y": y, "t": t, "x": x, "p1": x, "p2": x})

        result_ols = feols("y ~ t + x", data=df)
        result_oriv = oriv("y ~ t", ["p1", "p2"], df, "x_hat", vcov="iid")

        # Should be very close (not exact due to stacking + dummies)
        np.testing.assert_allclose(
            result_oriv.coef()["t"], result_ols.coef()["t"], rtol=0.01
        )
        np.testing.assert_allclose(
            result_oriv.coef()["x_hat"], result_ols.coef()["x"], rtol=0.01
        )
