from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.post_estimation.sensitivity import (
    SensitivityStatistics,
    compute_adjusted_estimate,
    compute_adjusted_se,
    compute_adjusted_t,
    compute_bias,
    compute_partial_f2,
    compute_partial_r2,
    compute_robustness_value,
)


@pytest.fixture
def sensitivity_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260826)
    n_obs = 240
    group = np.repeat(np.arange(24), n_obs // 24)
    benchmark = rng.normal(size=n_obs)
    control = rng.normal(size=n_obs)
    raw_positive = np.exp(0.2 * control + rng.normal(scale=0.4, size=n_obs))
    treatment = 0.45 * benchmark + 0.25 * control + rng.normal(size=n_obs)
    instrument = treatment + rng.normal(scale=0.25, size=n_obs)
    group_effect = rng.normal(scale=0.7, size=24)[group]
    outcome = (
        1.0
        + 0.8 * treatment
        + 0.55 * benchmark
        - 0.3 * control
        + group_effect
        + rng.normal(size=n_obs)
    )
    return pd.DataFrame(
        {
            "outcome": outcome,
            "treatment": treatment,
            "benchmark": benchmark,
            "control": control,
            "raw_positive": raw_positive,
            "instrument": instrument,
            "group": group,
            "weights": rng.uniform(0.5, 1.5, size=n_obs),
            "count_outcome": rng.poisson(np.exp(0.2 + 0.1 * treatment)),
        }
    )


def test_scalar_numerical_definitions():
    t_statistic = 2.5
    dof = 97

    partial_r2 = compute_partial_r2(t_statistic, dof)
    partial_f2 = compute_partial_f2(t_statistic, dof)

    assert partial_r2 == pytest.approx(t_statistic**2 / (t_statistic**2 + dof))
    assert partial_f2 == pytest.approx(t_statistic**2 / dof)
    assert partial_f2 == pytest.approx(partial_r2 / (1 - partial_r2))


def test_robustness_value_low_dof_branch_is_zero_not_negative():
    # This input exercised a negative-value bug in the original implementation.
    assert compute_robustness_value(3.0, 2, q=1, alpha=0.05) == 0


def test_vector_numerical_definitions_and_adjustments():
    r2dz_x = np.array([0.0, 0.1, 0.2])
    r2yz_dx = np.array([0.0, 0.2, 0.3])
    estimate = 1.2
    standard_error = 0.2
    dof = 100

    bias = compute_bias(r2dz_x, r2yz_dx, standard_error, dof)
    reduced = compute_adjusted_estimate(
        r2dz_x, r2yz_dx, estimate, standard_error, dof, reduce=True
    )
    increased = compute_adjusted_estimate(
        r2dz_x, r2yz_dx, estimate, standard_error, dof, reduce=False
    )
    adjusted_se = compute_adjusted_se(r2dz_x, r2yz_dx, standard_error, dof)
    adjusted_t = compute_adjusted_t(
        r2dz_x,
        r2yz_dx,
        estimate,
        standard_error,
        dof,
        reduce=True,
        h0=0.25,
    )

    np.testing.assert_allclose(reduced, estimate - bias)
    np.testing.assert_allclose(increased, estimate + bias)
    np.testing.assert_allclose(adjusted_t, (reduced - 0.25) / adjusted_se)


def test_analysis_uses_iid_statistics(sensitivity_data):
    fit = pf.feols(
        "outcome ~ treatment + benchmark + control", sensitivity_data, vcov="iid"
    )
    analysis = fit.sensitivity_analysis("treatment")
    stats = analysis.sensitivity_stats(q=1, alpha=0.05)
    index = fit._coefnames.index("treatment")

    assert isinstance(stats, SensitivityStatistics)
    assert stats.estimate == pytest.approx(fit._beta_hat[index])
    assert stats.standard_error == pytest.approx(fit._se[index])
    assert stats.degrees_of_freedom == fit._df_t
    assert stats.partial_r2 == pytest.approx(
        fit._tstat[index] ** 2 / (fit._tstat[index] ** 2 + fit._df_t)
    )
    assert stats.partial_f2 == pytest.approx(stats.partial_r2 / (1 - stats.partial_r2))
    assert 0 <= stats.robustness_value_alpha <= stats.robustness_value <= 1
    assert stats.to_dict()["estimate"] == stats.estimate

    with pytest.raises(FrozenInstanceError):
        stats.estimate = 0  # type: ignore[misc]


def test_treatment_is_required_and_retained(sensitivity_data):
    fit = pf.feols("outcome ~ treatment + benchmark", sensitivity_data)

    with pytest.raises(TypeError, match="treatment must be a coefficient name"):
        fit.sensitivity_analysis(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-intercept"):
        fit.sensitivity_analysis("Intercept")
    with pytest.raises(ValueError, match="not found"):
        fit.sensitivity_analysis("missing")

    analysis = fit.sensitivity_analysis("treatment")
    assert analysis.treatment == "treatment"
    assert analysis.partial_r2() == analysis.partial_r2("treatment")


def test_non_iid_fit_warns_and_matches_iid_analysis(sensitivity_data):
    iid_fit = pf.feols(
        "outcome ~ treatment + benchmark + control", sensitivity_data, vcov="iid"
    )
    robust_fit = pf.feols(
        "outcome ~ treatment + benchmark + control",
        sensitivity_data,
        vcov="hetero",
    )

    iid_stats = iid_fit.sensitivity_analysis("treatment").sensitivity_stats()
    with pytest.warns(UserWarning, match="uses IID standard errors"):
        robust_analysis = robust_fit.sensitivity_analysis("treatment")
    robust_stats = robust_analysis.sensitivity_stats()

    assert robust_fit._se[robust_fit._coefnames.index("treatment")] != pytest.approx(
        iid_stats.standard_error
    )
    assert robust_stats.to_dict() == pytest.approx(iid_stats.to_dict())


def test_bounds_use_prepared_matrices_with_fixed_effects_and_no_data_storage(
    sensitivity_data,
):
    fit = pf.feols(
        "outcome ~ treatment + benchmark + np.log(raw_positive) | group",
        sensitivity_data,
        store_data=False,
    )
    assert not hasattr(fit, "_data")

    analysis = fit.sensitivity_analysis("treatment")
    bounds = analysis.ovb_bounds(
        ["benchmark", "np.log(raw_positive)"], kd=[0.5, 1], ky=[1, 2]
    )

    assert len(bounds) == 4
    assert set(bounds["benchmark_covariate"]) == {
        "benchmark",
        "np.log(raw_positive)",
    }
    assert set(bounds["treatment"]) == {"treatment"}
    assert bounds["r2dz_x"].between(0, 1, inclusive="left").all()
    assert bounds["r2yz_dx"].between(0, 1, inclusive="both").all()
    assert {
        "adjusted_estimate",
        "adjusted_se",
        "adjusted_t",
        "adjusted_lower_ci",
        "adjusted_upper_ci",
    }.issubset(bounds)


def test_bounds_propagate_reduce_and_h0(sensitivity_data):
    fit = pf.feols("outcome ~ treatment + benchmark + control", sensitivity_data)
    analysis = fit.sensitivity_analysis("treatment")

    reduced = analysis.ovb_bounds("benchmark", reduce=True, h0=0.2)
    increased = analysis.ovb_bounds("benchmark", reduce=False, h0=0.2)

    assert abs(reduced.loc[0, "adjusted_estimate"]) < abs(
        increased.loc[0, "adjusted_estimate"]
    )
    expected_t = analysis.adjusted_t(
        reduced["r2dz_x"].to_numpy(),
        reduced["r2yz_dx"].to_numpy(),
        reduce=True,
        h0=0.2,
    )
    np.testing.assert_allclose(reduced["adjusted_t"], expected_t)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kd": [-1]}, "nonnegative"),
        ({"kd": [1, 2], "ky": [1]}, "same length"),
        ({"kd": []}, "must not be empty"),
    ],
)
def test_bounds_validate_multipliers(sensitivity_data, kwargs, message):
    fit = pf.feols("outcome ~ treatment + benchmark", sensitivity_data)
    analysis = fit.sensitivity_analysis("treatment")

    with pytest.raises(ValueError, match=message):
        analysis.ovb_bounds("benchmark", **kwargs)


def test_bounds_validate_benchmarks(sensitivity_data):
    fit = pf.feols("outcome ~ treatment + benchmark", sensitivity_data)
    analysis = fit.sensitivity_analysis("treatment")

    with pytest.raises(ValueError, match="must not be empty"):
        analysis.ovb_bounds([])
    with pytest.raises(ValueError, match="cannot also be the treatment"):
        analysis.ovb_bounds("treatment")
    with pytest.raises(ValueError, match="intercept"):
        analysis.ovb_bounds("Intercept")
    with pytest.raises(ValueError, match="not found"):
        analysis.ovb_bounds("missing")


@pytest.mark.parametrize(
    "function, args, message",
    [
        (compute_partial_r2, (1.0, 1), "at least 2"),
        (compute_robustness_value, (1.0, 10, -1, 0.05), "q must be nonnegative"),
        (compute_robustness_value, (1.0, 10, 1, 1.1), "alpha"),
        (compute_bias, (1.0, 0.2, 0.1, 10), "r2dz_x"),
        (compute_bias, (0.2, 1.1, 0.1, 10), "r2yz_dx"),
    ],
)
def test_numerical_functions_validate_inputs(function, args, message):
    with pytest.raises(ValueError, match=message):
        function(*args)


def test_summary_names_configured_treatment(sensitivity_data, capsys):
    fit = pf.feols("outcome ~ treatment + benchmark", sensitivity_data)
    analysis = fit.sensitivity_analysis("treatment")

    analysis.summary(benchmark_covariates="benchmark", q=0.5, reduce=False)

    output = capsys.readouterr().out
    assert "Treatment: treatment" in output
    assert "q = 0.5, reduce = False" in output
    assert "Bounds on omitted variable bias" in output


def test_store_data_false_is_supported_without_fixed_effects(sensitivity_data):
    fit = pf.feols(
        "outcome ~ treatment + benchmark",
        sensitivity_data,
        store_data=False,
    )

    stats = fit.sensitivity_analysis("treatment").sensitivity_stats()
    assert np.isfinite(stats.robustness_value)


def test_weighted_model_is_explicitly_unsupported(sensitivity_data):
    fit = pf.feols(
        "outcome ~ treatment + benchmark", sensitivity_data, weights="weights"
    )

    with pytest.raises(NotImplementedError, match="weighted feols"):
        fit.sensitivity_analysis("treatment")


def test_lean_model_is_explicitly_unsupported(sensitivity_data):
    fit = pf.feols("outcome ~ treatment + benchmark", sensitivity_data, lean=True)

    with pytest.raises(NotImplementedError, match="lean=True"):
        fit.sensitivity_analysis("treatment")


def test_iv_model_is_explicitly_unsupported(sensitivity_data):
    fit = pf.feols("outcome ~ benchmark + [treatment ~ instrument]", sensitivity_data)

    with pytest.raises(NotImplementedError, match="non-IV feols"):
        fit.sensitivity_analysis("treatment")


def test_poisson_model_is_explicitly_unsupported(sensitivity_data):
    fit = pf.fepois(
        "count_outcome ~ treatment + benchmark", sensitivity_data, vcov="iid"
    )

    with pytest.raises(NotImplementedError, match="non-IV feols"):
        fit.sensitivity_analysis("treatment")


def test_quantile_model_is_explicitly_unsupported(sensitivity_data):
    with pytest.warns(FutureWarning):
        fit = pf.quantreg("outcome ~ treatment + benchmark", sensitivity_data)

    with pytest.raises(NotImplementedError, match="non-IV feols"):
        fit.sensitivity_analysis("treatment")


def test_multiple_estimation_guides_user_to_fetch_model(sensitivity_data):
    fits = pf.feols("outcome ~ sw(treatment, benchmark)", sensitivity_data)

    with pytest.raises(NotImplementedError, match=r"Call fetch_model\(\) first"):
        fits.sensitivity_analysis("treatment")


def test_model_with_too_few_residual_degrees_of_freedom_is_rejected():
    data = pd.DataFrame(
        {
            "outcome": [1.0, 2.0, 4.0],
            "treatment": [0.0, 1.0, 2.0],
        }
    )
    fit = pf.feols("outcome ~ treatment", data)

    with pytest.raises(ValueError, match="at least 2 residual degrees"):
        fit.sensitivity_analysis("treatment")
