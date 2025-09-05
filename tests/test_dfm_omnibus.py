"""Tests for the DFM omnibus test for treatment effect heterogeneity."""

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf


class TestDFMOmnibus:
    """Test class for DFM omnibus test functionality."""

    @pytest.fixture
    def heterogeneous_data(self):
        """Create synthetic data with heterogeneous treatment effects."""
        np.random.seed(123)
        n = 1000
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        treatment = np.random.binomial(1, 0.5, n)

        # Heterogeneous treatment effect: varies with x1
        y = 2 + x1 + x2 + treatment * (1 + 0.5 * x1) + np.random.randn(n)

        return pd.DataFrame({"y": y, "treatment": treatment, "x1": x1, "x2": x2})

    @pytest.fixture
    def homogeneous_data(self):
        """Create synthetic data with homogeneous treatment effects."""
        np.random.seed(456)
        n = 1000
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        treatment = np.random.binomial(1, 0.5, n)

        # Homogeneous treatment effect: constant across x1 and x2
        y = 2 + x1 + x2 + treatment * 1.0 + np.random.randn(n)

        return pd.DataFrame({"y": y, "treatment": treatment, "x1": x1, "x2": x2})

    def test_dfm_test_basic_functionality(self, heterogeneous_data):
        """Test basic functionality of DFM test."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        result = fit.dfm_test(treatment_vars="treatment", interaction_vars=["x1", "x2"])

        # Check that result has expected structure
        assert isinstance(result, pd.Series)
        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result
        assert "distribution" in result
        assert "treatment_vars" in result
        assert "interaction_vars" in result
        assert "n_interactions_tested" in result

        # Check values make sense
        assert result["df"] == 2  # Two interaction terms
        assert result["distribution"] == "chi2"
        assert result["treatment_vars"] == ["treatment"]
        assert result["interaction_vars"] == ["x1", "x2"]
        assert result["n_interactions_tested"] == 2
        assert result["pvalue"] >= 0 and result["pvalue"] <= 1
        assert result["statistic"] >= 0

    def test_dfm_test_auto_detection(self, heterogeneous_data):
        """Test auto-detection of treatment variables."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        result = fit.dfm_test()

        # Should auto-detect 'treatment' as treatment variable
        assert result["treatment_vars"] == ["treatment"]
        assert set(result["interaction_vars"]) == {"x1", "x2"}

    def test_dfm_test_heterogeneous_vs_homogeneous(
        self, heterogeneous_data, homogeneous_data
    ):
        """Test that the test can distinguish heterogeneous from homogeneous effects."""
        # Test with heterogeneous data - should reject null
        fit_het = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)
        result_het = fit_het.dfm_test()

        # Test with homogeneous data - should not reject null
        fit_hom = pf.feols("y ~ treatment + x1 + x2", data=homogeneous_data)
        result_hom = fit_hom.dfm_test()

        # Heterogeneous data should have smaller p-value
        assert result_het["pvalue"] < result_hom["pvalue"]

        # With strong heterogeneity, should strongly reject null
        assert result_het["pvalue"] < 0.05

    def test_dfm_test_f_distribution(self, heterogeneous_data):
        """Test DFM test with F distribution."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        result = fit.dfm_test(distribution="F")

        assert result["distribution"] == "F"
        assert "statistic" in result
        assert "pvalue" in result

    def test_dfm_test_multiple_treatments(self, heterogeneous_data):
        """Test DFM test with multiple treatment variables."""
        # Add a second treatment variable
        heterogeneous_data["treatment2"] = np.random.binomial(
            1, 0.3, len(heterogeneous_data)
        )

        fit = pf.feols("y ~ treatment + treatment2 + x1 + x2", data=heterogeneous_data)

        result = fit.dfm_test(treatment_vars=["treatment", "treatment2"])

        assert set(result["treatment_vars"]) == {"treatment", "treatment2"}
        assert result["n_interactions_tested"] == 4  # 2 treatments × 2 covariates

    def test_dfm_test_single_interaction_var(self, heterogeneous_data):
        """Test DFM test with single interaction variable."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        result = fit.dfm_test(interaction_vars="x1")

        assert result["interaction_vars"] == ["x1"]
        assert result["n_interactions_tested"] == 1

    def test_dfm_test_error_no_treatment_detected(self):
        """Test error when no treatment variables can be detected."""
        np.random.seed(789)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = x1 + x2 + np.random.randn(n)

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        fit = pf.feols("y ~ x1 + x2", data=data)

        with pytest.raises(
            ValueError, match="Could not automatically detect treatment variables"
        ):
            fit.dfm_test()

    def test_dfm_test_error_invalid_treatment_var(self, heterogeneous_data):
        """Test error when specified treatment variable not in model."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        with pytest.raises(
            ValueError, match="Treatment variable 'nonexistent' not found"
        ):
            fit.dfm_test(treatment_vars="nonexistent")

    def test_dfm_test_error_invalid_interaction_var(self, heterogeneous_data):
        """Test error when interaction variable not in data."""
        fit = pf.feols("y ~ treatment + x1 + x2", data=heterogeneous_data)

        with pytest.raises(
            ValueError, match="Interaction variable 'nonexistent' not found"
        ):
            fit.dfm_test(interaction_vars="nonexistent")

    def test_dfm_test_error_no_interaction_vars(self):
        """Test error when no interaction variables available."""
        np.random.seed(999)
        n = 100
        treatment = np.random.binomial(1, 0.5, n)
        y = 2 + treatment + np.random.randn(n)

        data = pd.DataFrame({"y": y, "treatment": treatment})
        fit = pf.feols("y ~ treatment", data=data)

        with pytest.raises(ValueError, match="No interaction variables available"):
            fit.dfm_test()

    def test_dfm_test_with_fixed_effects(self, heterogeneous_data):
        """Test DFM test works with fixed effects."""
        # Add a group variable for fixed effects
        heterogeneous_data["group"] = np.random.choice(
            range(10), len(heterogeneous_data)
        )

        fit = pf.feols("y ~ treatment + x1 + x2 | group", data=heterogeneous_data)

        result = fit.dfm_test()

        # Should work and detect heterogeneity
        assert isinstance(result, pd.Series)
        assert result["pvalue"] < 0.05

    def test_dfm_test_different_naming_patterns(self):
        """Test auto-detection with different treatment variable naming patterns."""
        np.random.seed(111)
        n = 500
        x1 = np.random.randn(n)

        test_cases = [
            ("treat", "treat"),
            ("policy", "policy"),
            ("intervention", "intervention"),
            ("trt", "trt"),
        ]

        for var_name, expected in test_cases:
            treatment = np.random.binomial(1, 0.5, n)
            y = 2 + x1 + treatment * (1 + 0.3 * x1) + np.random.randn(n)

            data = pd.DataFrame({"y": y, var_name: treatment, "x1": x1})
            fit = pf.feols(f"y ~ {var_name} + x1", data=data)

            result = fit.dfm_test()
            assert expected in result["treatment_vars"]
