import pytest
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.options import get_option, set_option, option_context

@pytest.fixture
def data():
    """Create test data."""
    np.random.seed(123)
    n = 100
    data = pd.DataFrame({
        'y': np.random.normal(0, 1, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'id': np.repeat(range(10), 10),
        'weights': np.random.uniform(0.5, 1.5, n)
    })
    return data

def test_default_options_feols(data):
    """Test that default options are correctly applied to feols."""
    # Get default options
    default_vcov = get_option('vcov')
    default_weights = get_option('weights')
    default_ssc = get_option('ssc')
    default_fixef_tol = get_option('fixef_tol')
    default_collin_tol = get_option('collin_tol')

    # Fit model with defaults
    fit = pf.feols('y ~ x1 + x2 | id', data=data)

    # Check that defaults were applied
    assert fit._vcov_type == default_vcov
    assert fit._weights is default_weights
    assert fit._ssc_dict == default_ssc
    assert fit._fixef_tol == default_fixef_tol
    assert fit._collin_tol == default_collin_tol

def test_default_options_fepois(data):
    """Test that default options are correctly applied to fepois."""
    # Get default options
    default_vcov = get_option('vcov')
    default_weights = get_option('weights')
    default_ssc = get_option('ssc')
    default_fixef_tol = get_option('fixef_tol')
    default_collin_tol = get_option('collin_tol')

    # Fit model with defaults
    fit = pf.fepois('y ~ x1 + x2 | id', data=data)

    # Check that defaults were applied
    assert fit._vcov_type == default_vcov
    assert fit._weights is None  # No weights by default
    assert fit._ssc_dict == default_ssc
    assert fit._fixef_tol == default_fixef_tol
    assert fit._collin_tol == default_collin_tol

def test_default_options_feglm(data):
    """Test that default options are correctly applied to feglm."""
    # Get default options
    default_vcov = get_option('vcov')
    default_weights = get_option('weights')
    default_ssc = get_option('ssc')
    default_fixef_tol = get_option('fixef_tol')
    default_collin_tol = get_option('collin_tol')

    # Fit model with defaults
    fit = pf.feglm('y ~ x1 + x2 | id', data=data)

    # Check that defaults were applied
    assert fit._vcov_type == default_vcov
    assert fit._weights is None  # No weights by default
    assert fit._ssc_dict == default_ssc
    assert fit._fixef_tol == default_fixef_tol
    assert fit._collin_tol == default_collin_tol

def test_override_options(data):
    """Test that options can be overridden at the function call level."""
    # Set custom options
    custom_vcov = "hetero"
    custom_weights = "weights"
    custom_ssc = {"adj": True, "fixef_k": "none"}

    # Fit model with custom options
    fit = pf.feols(
        'y ~ x1 + x2 | id',
        data=data,
        vcov=custom_vcov,
        weights=custom_weights,
        ssc=custom_ssc
    )

    # Check that custom options were applied
    assert fit._vcov_type == custom_vcov
    assert fit._weights_name == custom_weights
    assert fit._ssc_dict == custom_ssc

def test_option_context(data):
    """Test that options can be temporarily changed using the context manager."""
    # Original options
    original_vcov = get_option('vcov')
    original_weights = get_option('weights')

    # Custom options
    custom_vcov = "hetero"
    custom_weights = "weights"

    # Use context manager to temporarily change options
    with option_context(vcov=custom_vcov, weights=custom_weights):
        fit = pf.feols('y ~ x1 + x2 | id', data=data)
        assert fit._vcov_type == custom_vcov
        assert fit._weights_name == custom_weights

    # Check that original options are restored
    assert get_option('vcov') == original_vcov
    assert get_option('weights') == original_weights

def test_set_option(data):
    """Test that options can be permanently changed using set_option."""
    # Original options
    original_vcov = get_option('vcov')
    original_weights = get_option('weights')

    try:
        # Set new options
        custom_vcov = "hetero"
        custom_weights = "weights"
        set_option(vcov=custom_vcov, weights=custom_weights)

        # Fit model with new defaults
        fit = pf.feols('y ~ x1 + x2 | id', data=data)
        assert fit._vcov_type == custom_vcov
        assert fit._weights_name == custom_weights

    finally:
        # Restore original options
        set_option(vcov=original_vcov, weights=original_weights)

def test_options_persistence(data):
    """Test that options persist across multiple function calls."""
    # Set custom options
    custom_vcov = "hetero"
    custom_weights = "weights"
    set_option(vcov=custom_vcov, weights=custom_weights)

    try:
        # Fit multiple models
        fit1 = pf.feols('y ~ x1 | id', data=data)
        fit2 = pf.fepois('y ~ x1 | id', data=data)
        fit3 = pf.feglm('y ~ x1 | id', data=data)

        # Check that all models use the custom options
        assert fit1._vcov_type == custom_vcov
        assert fit2._vcov_type == custom_vcov
        assert fit3._vcov_type == custom_vcov

        assert fit1._weights_name == custom_weights
        assert fit2._weights_name == custom_weights
        assert fit3._weights_name == custom_weights

    finally:
        # Restore original options
        set_option(vcov=get_option('vcov'), weights=get_option('weights'))

def test_global_vs_direct_options(data):
    """Test that applying global options leads to the same results as providing arguments directly."""
    # Set custom options
    custom_vcov = "hetero"
    custom_weights = "weights"
    custom_ssc = {"adj": True, "fixef_k": "none"}

    # Fit model with direct arguments
    direct_fit = pf.feols(
        'y ~ x1 + x2 | id',
        data=data,
        vcov=custom_vcov,
        weights=custom_weights,
        ssc=custom_ssc
    )

    # Set global options
    set_option(vcov=custom_vcov, weights=custom_weights, ssc=custom_ssc)

    try:
        # Fit model with global options
        global_fit = pf.feols('y ~ x1 + x2 | id', data=data)

        # Compare coefficients
        np.testing.assert_allclose(
            direct_fit.coef(),
            global_fit.coef(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Coefficients do not match between direct and global options"
        )

        # Compare standard errors
        np.testing.assert_allclose(
            direct_fit.se(),
            global_fit.se(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Standard errors do not match between direct and global options"
        )

        # Compare t-statistics
        np.testing.assert_allclose(
            direct_fit.tstat(),
            global_fit.tstat(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="t-statistics do not match between direct and global options"
        )

        # Compare p-values
        np.testing.assert_allclose(
            direct_fit.pvalue(),
            global_fit.pvalue(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="p-values do not match between direct and global options"
        )

        # Compare confidence intervals
        np.testing.assert_allclose(
            direct_fit.confint().values,
            global_fit.confint().values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Confidence intervals do not match between direct and global options"
        )

        # Compare variance-covariance matrices
        np.testing.assert_allclose(
            direct_fit._vcov,
            global_fit._vcov,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Variance-covariance matrices do not match between direct and global options"
        )

        # Compare predictions
        np.testing.assert_allclose(
            direct_fit.predict(),
            global_fit.predict(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Predictions do not match between direct and global options"
        )

        # Compare residuals
        np.testing.assert_allclose(
            direct_fit.resid(),
            global_fit.resid(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="Residuals do not match between direct and global options"
        )

    finally:
        # Restore original options
        set_option(vcov=get_option('vcov'), weights=get_option('weights'), ssc=get_option('ssc'))