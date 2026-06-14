import numpy as np
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


@pytest.mark.parametrize("fml", fml_ols_vs_gaussian)
@pytest.mark.parametrize("inference", ["iid", "hetero", {"CRV1": "f1"}])
@pytest.mark.parametrize("dropna", [True])
def test_ols_vs_gaussian_glm(fml, inference, dropna):
    data = pf.get_data()
    if dropna:
        data = data.dropna()

    fit_ols = pf.feols(fml=fml, data=data, vcov=inference)
    fit_gaussian = pf.feglm(fml=fml, data=data, family="gaussian", vcov=inference)

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
