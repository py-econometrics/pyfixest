import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import NonConvergenceError
from pyfixest.estimation.internals import fit_glm_ as fit_glm_module
from pyfixest.estimation.internals.families import POISSON


def check_absolute_diff(x1, x2, tol, msg=None):
    "Check for absolute differences."
    if isinstance(x1, (int, float)):
        x1 = np.array([x1])
    if isinstance(x2, (int, float)):
        x2 = np.array([x2])
        msg = "" if msg is None else msg

    # handle nan values
    nan_mask_x1 = np.isnan(x1)
    nan_mask_x2 = np.isnan(x2)

    if not np.array_equal(nan_mask_x1, nan_mask_x2):
        raise AssertionError(f"{msg}: NaN positions do not match")

    valid_mask = ~nan_mask_x1  # Mask for non-NaN elements (same for x1 and x2)
    assert np.all(np.abs(x1[valid_mask] - x2[valid_mask]) < tol), msg


fml_list = [
    ("Y ~ X1 + X2 + C(f1)", "Y ~ X1 + X2 | f1"),
    ("Y ~ X1 + X2 + C(f1) + C(f2)", "Y ~ X1 + X2 | f1 + f2"),
]

fml_ols_vs_gaussian = ["Y ~ X1", "Y ~ X1 + C(f1)", "Y ~ X1 * X2"]


@pytest.mark.parametrize("family", ["gaussian", "logit", "probit", "poisson"])
def test_glm_keeps_formula_observation_and_working_domains_distinct(family):
    """Retain formula inputs and observation weights beside final IWLS state."""
    rng = np.random.default_rng(918273)
    n_groups = 12
    group_size = 15
    n_obs = n_groups * group_size
    fixed_effect = np.repeat(np.arange(n_groups), group_size)
    covariate = rng.normal(size=n_obs)
    linear_predictor = -0.2 + 0.7 * covariate + 0.05 * fixed_effect

    if family == "gaussian":
        response = linear_predictor + rng.normal(scale=0.5, size=n_obs)
    elif family == "poisson":
        # Strict positivity prevents a group from being removed for separation.
        response = rng.poisson(np.exp(linear_predictor)) + 1
    else:
        probability = 1 / (1 + np.exp(-linear_predictor))
        response = rng.binomial(1, probability)

    observation_weights = np.linspace(0.5, 2.0, n_obs)
    data = pd.DataFrame(
        {
            "y": response,
            "x": covariate,
            "fe": fixed_effect,
            "weight": observation_weights,
        }
    )
    fit = pf.feglm(
        "y ~ x | fe",
        data=data,
        family=family,
        weights="weight",
        vcov="hetero",
        separation_check=[],
        iwls_tol=1e-10,
    )

    assert isinstance(fit._formula_data.dependent, pd.DataFrame)
    assert isinstance(fit._formula_data.independent, pd.DataFrame)
    assert isinstance(fit._fe, pd.DataFrame)
    np.testing.assert_allclose(
        fit._observation_weights.values,
        observation_weights,
    )
    np.testing.assert_allclose(fit._weights.flatten(), observation_weights)

    working = fit._working_state
    assert fit._X is working.design_within
    assert fit._Y is working.working_response_within
    assert fit._Z is working.design_within
    assert fit._irls_weights is working.working_weights
    assert not hasattr(working, "sqrt_working_weights")
    assert not hasattr(working, "design_solver")
    assert not hasattr(working, "response_solver")

    np.testing.assert_allclose(fit.resid("response"), working.response_residuals)
    np.testing.assert_allclose(fit.resid("working"), working.working_residuals)
    np.testing.assert_allclose(
        fit._scores,
        working.design_within
        * (working.working_weights * working.working_residuals)[:, None],
    )
    expected_hessian = working.design_within.T @ (
        working.working_weights[:, None] * working.design_within
    )
    np.testing.assert_allclose(fit._hessian, expected_hessian)
    np.testing.assert_allclose(fit._leverage_weights(), working.working_weights)
    np.testing.assert_allclose(fit._fixef_weights(), working.working_weights)

    for group in np.unique(fixed_effect):
        group_rows = fixed_effect == group
        group_weights = working.working_weights[group_rows]
        np.testing.assert_allclose(
            group_weights @ working.design_within[group_rows],
            0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            group_weights @ working.working_response_within[group_rows],
            0,
            atol=1e-8,
        )

    if family == "gaussian":
        np.testing.assert_allclose(working.working_weights, observation_weights)
    else:
        assert not np.allclose(working.working_weights, observation_weights)


@pytest.mark.parametrize("vcov", ["HC2", "HC3"])
def test_feglm_frequency_weight_leverage_matches_expanded_sample(vcov):
    """Use observation counts, not IRLS weights, in fweight HC leverage."""
    rng = np.random.default_rng(102938)
    covariate = np.linspace(-1.5, 1.5, 80)
    probability = 1 / (1 + np.exp(-(-0.15 + 0.8 * covariate)))
    aggregated = pd.DataFrame(
        {
            "y": rng.binomial(1, probability),
            "x": covariate,
            "count": rng.integers(1, 5, size=len(covariate)),
        }
    )
    expanded = aggregated.loc[aggregated.index.repeat(aggregated["count"])].copy()

    fit_frequency = pf.feglm(
        "y ~ x",
        data=aggregated,
        family="logit",
        weights="count",
        weights_type="fweights",
        vcov=vcov,
        iwls_tol=1e-11,
    )
    fit_expanded = pf.feglm(
        "y ~ x",
        data=expanded,
        family="logit",
        vcov=vcov,
        iwls_tol=1e-11,
    )

    np.testing.assert_allclose(
        fit_frequency.coef(),
        fit_expanded.coef(),
        atol=1e-10,
        err_msg="Frequency-weighted logit coefficients differ from row expansion.",
    )
    # Collapsing duplicate rows changes floating-point accumulation order and
    # therefore the final IWLS stopping point at roughly the low 1e-6 scale.
    np.testing.assert_allclose(
        fit_frequency._vcov,
        fit_expanded._vcov,
        atol=1e-9,
        rtol=5e-6,
        err_msg=f"Frequency-weighted logit {vcov} differs from row expansion.",
    )


def test_feglm_frequency_weight_sample_size_survives_separation():
    """Recompute effective and physical sample sizes in their distinct domains."""
    aggregated = pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            "x": [-1.2, -0.4, 0.7, -1.1, -0.2, 0.9, -0.8, 0.1, 1.2, -0.9, 0.4, 1.4],
            "fe": np.repeat(list("abcd"), 3),
            "count": [1, 3, 2, 2, 1, 4, 1, 2, 3, 3, 2, 1],
        }
    )
    expanded = aggregated.loc[aggregated.index.repeat(aggregated["count"])].copy()

    with pytest.warns(UserWarning, match="observations removed because of separation"):
        fit_frequency = pf.feglm(
            "y ~ x | fe",
            data=aggregated,
            family="logit",
            weights="count",
            weights_type="fweights",
            vcov="HC1",
            separation_check=["fe"],
            iwls_tol=1e-11,
        )
    with pytest.warns(UserWarning, match="observations removed because of separation"):
        fit_expanded = pf.feglm(
            "y ~ x | fe",
            data=expanded,
            family="logit",
            vcov="HC1",
            separation_check=["fe"],
            iwls_tol=1e-11,
        )

    assert fit_frequency._N == fit_expanded._N == 19
    assert fit_frequency._N_rows == 9
    assert fit_expanded._N_rows == 19
    assert fit_frequency._observation_weights.n_effective == 19
    np.testing.assert_allclose(fit_frequency.coef(), fit_expanded.coef(), atol=1e-10)
    np.testing.assert_allclose(fit_frequency._vcov, fit_expanded._vcov, atol=1e-10)


@pytest.mark.parametrize("fml", fml_ols_vs_gaussian)
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "f1"}])
@pytest.mark.parametrize("dropna", [True])
@pytest.mark.parametrize("weights", [None, "weights"])
def test_ols_vs_gaussian_glm(fml, inference, dropna, weights):
    data = pf.get_data()
    if dropna:
        data = data.dropna()

    fit_ols = pf.feols(fml=fml, data=data, vcov=inference, weights=weights)
    fit_gaussian = pf.feglm(
        fml=fml,
        data=data,
        family="gaussian",
        vcov=inference,
        weights=weights,
    )

    check_absolute_diff(
        fit_ols.coef().xs("X1"), fit_gaussian.coef().xs("X1"), tol=1e-10
    )
    check_absolute_diff(fit_ols._weights[0:5], fit_gaussian._weights[0:5], tol=1e-10)
    check_absolute_diff(fit_ols._u_hat[0:5], fit_gaussian._u_hat[0:5], tol=1e-10)
    check_absolute_diff(fit_ols._scores[0, :], fit_gaussian._scores[0, :], tol=1e-10)

    if inference == "iid":
        # iid inference different: follows iid-glm; just the bread and not bread x sigma2
        scaling_factor = fit_ols._vcov[0, 0] / fit_gaussian._vcov[0, 0]
        # Check that all elements follow the same scaling
        check_absolute_diff(
            fit_ols._vcov, scaling_factor * fit_gaussian._vcov, tol=1e-10
        )
    else:
        check_absolute_diff(fit_ols._vcov, fit_gaussian._vcov, tol=1e-10)


@pytest.mark.parametrize("fml", fml_list)
@pytest.mark.parametrize("family", ["gaussian", "logit", "probit"])
def test_glm_fe_vs_onehot(fml, family):
    """
    Test that GLM with fixed effects produces the same coefficients and SEs
    as GLM with one-hot encoded fixed effects (C(fe) syntax).
    """
    data = pf.get_data()
    if family in ["logit", "probit"]:
        data["Y"] = np.where(data["Y"] > 0, 1, 0)

    fml_onehot, fml_fe = fml

    fit_onehot = pf.feglm(
        fml=fml_onehot, data=data, family=family, ssc=pf.ssc(k_adj=False, G_adj=False)
    )
    fit_fe = pf.feglm(
        fml=fml_fe, data=data, family=family, ssc=pf.ssc(k_adj=False, G_adj=False)
    )

    for coef_name in ["X1", "X2"]:
        check_absolute_diff(
            fit_onehot.coef().xs(coef_name),
            fit_fe.coef().xs(coef_name),
            1e-08,
            f"Coef {coef_name} mismatch for fml={fml} and family={family}",
        )
        check_absolute_diff(
            fit_onehot.se().xs(coef_name),
            fit_fe.se().xs(coef_name),
            1e-08,
            f"SE {coef_name} mismatch for fml={fml} and family={family}",
        )


def test_step_halving_forces_follow_up_wls(monkeypatch):
    """Do not declare convergence immediately after accepting a shortened step."""
    x = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([np.ones_like(x), x])
    Y = np.array(
        [
            0,
            1,
            0,
            2,
            1,
            3,
            0,
            2,
            1,
            4,
            2,
            3,
            1,
            2,
            4,
            5,
            3,
            4,
            2,
            6,
            4,
            5,
            3,
            7,
            4,
            6,
            5,
            8,
            6,
            9,
        ],
        dtype=float,
    )

    demean_calls = 0

    def _identity_demean(v, X, weights, tol):
        nonlocal demean_calls
        demean_calls += 1
        return v, X

    step_calls = 0

    def _fake_step_halving(
        family,
        y_flat,
        eta,
        eta_new,
        mu_new,
        deviance,
        deviance_new,
        tol,
        weights,
        step_halving_tol=1e-12,
    ):
        nonlocal step_calls
        step_calls += 1
        if step_calls == 1:
            eta_accepted = eta + 0.5 * (eta_new - eta)
            mu_accepted = family.inv_link(eta_accepted)
            return eta_accepted, mu_accepted, deviance - 1e-12, True
        return eta_new, mu_new, deviance - 1e-12, False

    monkeypatch.setattr(fit_glm_module, "_step_halving", _fake_step_halving)

    fit_glm_module.fit_glm_irls(
        X=X,
        Y=Y,
        family=POISSON,
        demean=_identity_demean,
        coefnames=["Intercept", "X"],
        collin_tol=1e-9,
        accelerate=False,
        maxiter=3,
        tol=1e-8,
    )

    assert step_calls == 2
    assert demean_calls == 3


def test_glm_raises_after_iwls_maxiter_without_convergence():
    """Exhausting maxiter should not return a silently unconverged fit."""
    x = np.linspace(-1.0, 1.0, 30)
    X = np.column_stack([np.ones_like(x), x])
    Y = np.array(
        [
            0,
            1,
            0,
            2,
            1,
            3,
            0,
            2,
            1,
            4,
            2,
            3,
            1,
            2,
            4,
            5,
            3,
            4,
            2,
            6,
            4,
            5,
            3,
            7,
            4,
            6,
            5,
            8,
            6,
            9,
        ],
        dtype=float,
    )

    def _identity_demean(v, X, weights, tol):
        return v, X

    with pytest.raises(NonConvergenceError, match="did not converge"):
        fit_glm_module.fit_glm_irls(
            X=X,
            Y=Y,
            family=POISSON,
            demean=_identity_demean,
            coefnames=["Intercept", "X"],
            collin_tol=1e-9,
            accelerate=False,
            maxiter=1,
            tol=1e-14,
        )
