import numpy as np
import pytest

import pyfixest as pf


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
