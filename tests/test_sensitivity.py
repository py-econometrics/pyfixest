import matplotlib
import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols, fepois
from pyfixest.utils.utils import get_data

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt


def test_partial_measures():
    """Test Partial R2 and Partial f2 calculations."""
    data = get_data()
    # Create a model: Y ~ X1 + X2
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()  # Or SensitivityAnalysis(fit)

    # 1. Scalar Case (Specific Variable)
    r2_x1 = sens.partial_r2("X1")
    f2_x1 = sens.partial_f2("X1")

    assert isinstance(r2_x1, (float, np.float64))
    assert 0 <= r2_x1 <= 1
    assert f2_x1 > 0
    # Check consistency: f2 = R2 / (1 - R2)
    assert np.isclose(f2_x1, r2_x1 / (1 - r2_x1))

    # 2. Vector Case (All Variables)
    r2_all = sens.partial_r2()
    assert len(r2_all) == 3  # Intercept, X1, X2
    assert isinstance(r2_all, np.ndarray)


def test_robustness_value_logic():
    """Test Robustness Value (RV) bounds and logic."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # RV should always be between 0 and 1
    rv = sens.robustness_value("X1", q=1, alpha=0.05)
    assert 0 <= rv <= 1

    # RV with alpha=1 (Point Estimate) should be >= RV with alpha=0.05 (Significance)
    rv_q = sens.robustness_value("X1", q=1, alpha=1)
    rv_qa = sens.robustness_value("X1", q=1, alpha=0.05)
    assert rv_q >= rv_qa


def test_sensitivity_stats():
    """Test the main dictionary summary function."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Case 1: Specific variable (Scalar output)
    stats_single = sens.sensitivity_stats(X="X1")
    assert isinstance(stats_single, dict)
    required_keys = ["estimate", "se", "partial_R2", "rv_q", "rv_qa"]
    for key in required_keys:
        assert key in stats_single

    # Case 2: All variables (Vector output)
    stats_all = sens.sensitivity_stats(X=None)
    assert isinstance(stats_all, dict)
    assert len(stats_all["estimate"]) == len(fit._coefnames)
    # Ensure RV is calculated for all
    assert len(stats_all["rv_q"]) == len(fit._coefnames)


def test_ovb_bounds_structure():
    """Test if ovb_bounds returns the correct DataFrame structure."""
    data = get_data()
    # Model: Y ~ X1 + X2
    # We will treat X1 as treatment, X2 as the benchmark
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    bounds = sens.ovb_bounds(
        treatment="X1", benchmark_covariates="X2", kd=[1, 2], ky=[1, 2]
    )

    # 1. Check Return Type
    assert isinstance(bounds, pd.DataFrame)

    # 2. Check Dimensions
    # We asked for kd=[1,2] for 1 benchmark ("X2"). Should be 2 rows.
    assert len(bounds) == 2

    # 3. Check Columns
    expected_cols = [
        "bound_label",
        "r2dz_x",
        "r2yz_dx",
        "treatment",
        "adjusted_estimate",
        "adjusted_se",
        "adjusted_t",
        "adjusted_lower_CI",
        "adjusted_upper_CI",
    ]
    for col in expected_cols:
        assert col in bounds.columns


def test_ovb_bounds_adjustment_logic():
    """Test if the adjusted estimates move in the expected direction."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Get original estimate for X1
    original_est = fit.coef()["X1"]

    # Calculate bounds with reduce=True (Default)
    # This should shrink the estimate towards zero
    bounds = sens.ovb_bounds(treatment="X1", benchmark_covariates="X2", kd=1)

    adj_est = bounds["adjusted_estimate"].iloc[0]

    # Assert magnitude reduction
    assert abs(adj_est) < abs(original_est)


def test_summary_smoke_test(capsys):
    """
    Smoke test for the summary print method.
    Uses capsys to suppress/capture stdout.
    """
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Should run without error
    sens.summary(treatment="X1", benchmark_covariates="X2")

    # Optional: Check if output contains expected strings
    captured = capsys.readouterr()
    assert "Sensitivity Analysis to Unobserved Confounding" in captured.out
    assert "Bounds on omitted variable bias" in captured.out
    assert "1x X2" in captured.out


def test_error_handling():
    """Ensure proper errors are raised for invalid inputs."""
    # 1. Setup Data with guaranteed correlation
    # We need X1 and X2 to be correlated so that r2dxj_x > 0.
    # If r2dxj_x is 0, no kd value will ever trigger the "Impossible" error.
    rng = np.random.default_rng(123)
    N = 100
    X2 = rng.normal(0, 1, N)
    # Make X1 explicitly dependent on X2 (Correlation ~ 0.5)
    X1 = 0.5 * X2 + rng.normal(0, 1, N)
    Y = X1 + X2 + rng.normal(0, 1, N)

    data = pd.DataFrame({"Y": Y, "X1": X1, "X2": X2})

    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # 2. Test Invalid Bound Type
    with pytest.raises(ValueError, match="Only partial r2 is implemented"):
        sens.ovb_bounds(treatment="X1", benchmark_covariates="X2", bound="invalid_type")

    # 3. Test Impossible kd value
    # Since X1 and X2 are correlated, a massive kd will imply R2 > 1
    with pytest.raises(ValueError, match="Impossible scenario"):
        sens.ovb_bounds(treatment="X1", benchmark_covariates="X2", kd=1000)


def test_sensitivity_analysis_feols_supported():
    """Test that sensitivity analysis works for Feols (OLS) models."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)

    # Should not raise
    sens = fit.sensitivity_analysis()
    assert sens is not None


def test_sensitivity_analysis_fepois_not_supported():
    """Test that sensitivity analysis raises error for Poisson models."""
    data = get_data()
    data = data.dropna()  # Remove NaN values first
    data["Y_count"] = np.abs(data["Y"]).astype(int) + 1

    fit = fepois("Y_count ~ X1 + X2", data=data)

    with pytest.raises(ValueError, match="only supported for OLS"):
        fit.sensitivity_analysis()


def test_sensitivity_analysis_feiv_not_supported():
    """Test that sensitivity analysis raises error for IV models."""
    data = get_data()
    data["Z1"] = data["X1"] + np.random.normal(0, 0.1, len(data))  # Instrument

    fit = feols("Y ~ 1 | X1 ~ Z1", data=data)

    with pytest.raises(ValueError, match="only supported for OLS"):
        fit.sensitivity_analysis()


@pytest.mark.extended
def test_contour_plot_basic():
    """Smoke test for contour plot with default parameters."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Basic contour plot - estimate sensitivity
    fig = sens.plot(treatment="X1", plot_type="contour")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_contour_plot_with_benchmarks():
    """Test contour plot with benchmark covariates."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Contour plot with benchmarks
    fig = sens.plot(
        treatment="X1", plot_type="contour", benchmark_covariates="X2", kd=[1, 2]
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_contour_plot_t_value():
    """Test contour plot with t-value sensitivity."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Contour plot with t-value sensitivity
    fig = sens.plot(treatment="X1", plot_type="contour", sensitivity_of="t-value")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_contour_plot_custom_params():
    """Test contour plot with various custom parameters."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Contour plot with custom parameters
    fig = sens.plot(
        treatment="X1",
        plot_type="contour",
        benchmark_covariates="X2",
        kd=[1, 2, 3],
        ky=[1, 2, 3],
        lim=0.5,
        lim_y=0.5,
        figsize=(8, 8),
        estimate_threshold=0,
        t_threshold=1.96,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_extreme_plot_basic():
    """Smoke test for extreme plot with default parameters."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Basic extreme plot
    fig = sens.plot(treatment="X1", plot_type="extreme")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_extreme_plot_with_benchmarks():
    """Test extreme plot with benchmark covariates."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Extreme plot with benchmarks
    fig = sens.plot(
        treatment="X1", plot_type="extreme", benchmark_covariates="X2", kd=[1, 2]
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_extreme_plot_custom_params():
    """Test extreme plot with custom parameters."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    # Extreme plot with custom r2yz_dx scenarios
    fig = sens.plot(
        treatment="X1",
        plot_type="extreme",
        r2yz_dx=[1.0, 0.75, 0.5, 0.25],
        figsize=(10, 6),
        threshold=0.5,
    )
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


@pytest.mark.extended
def test_plot_invalid_type():
    """Test that invalid plot_type raises ValueError."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    with pytest.raises(ValueError, match='plot_type must be "contour" or "extreme"'):
        sens.plot(treatment="X1", plot_type="invalid")


@pytest.mark.extended
def test_extreme_plot_t_value_not_implemented():
    """Test that extreme plot with t-value raises NotImplementedError."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    with pytest.raises(
        NotImplementedError, match="Extreme sensitivity plots for t-values"
    ):
        sens.plot(treatment="X1", plot_type="extreme", sensitivity_of="t-value")


@pytest.mark.extended
def test_contour_plot_invalid_sensitivity_of():
    """Test that invalid sensitivity_of raises ValueError in contour plot."""
    data = get_data()
    fit = feols("Y ~ X1 + X2", data=data)
    sens = fit.sensitivity_analysis()

    with pytest.raises(ValueError, match="sensitivity_of must be either"):
        sens.plot(treatment="X1", plot_type="contour", sensitivity_of="invalid")
