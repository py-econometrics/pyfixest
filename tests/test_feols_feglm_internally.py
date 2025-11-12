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
    ("Y ~ X1 + C(f1)", "Y~X1 | f1"),
    ("Y ~ X1 + C(f1) + C(f2)", "Y~X1 | f1 + f2"),
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

    # iid inference different: follows iid-glm; just the bread and not bread x sigma2
    check_absolute_diff(fit_ols.se().xs("X1"), fit_gaussian.se().xs("X1"), tol=1e-10)
    check_absolute_diff(fit_ols._vcov[0, 0], fit_gaussian._vcov[0, 0], tol=1e-10)


@pytest.mark.skip("Fixed effects are not yet supported.")
@pytest.mark.parametrize("fml", fml_list)
@pytest.mark.parametrize("family", ["gaussian"])
def test_feols_feglm_internally(fml, family):
    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > 0, 1, 0)

    fml1, fml2 = fml

    fit1 = pf.feglm(
        fml=fml1, data=data, family=family, ssc=pf.ssc(k_adj=False, G_adj=False)
    )
    fit2 = pf.feglm(
        fml=fml2, data=data, family=family, ssc=pf.ssc(k_adj=False, G_adj=False)
    )

    assert fit1.coef().xs("X1") == fit2.coef().xs("X1"), (
        f"Test failed for fml = {fml} and family = gaussian"
    )
    assert fit1.se().xs("X1") == fit2.se().xs("X1"), (
        f"Test failed for fml = {fml} and family = gaussian"
    )
    assert fit1._u_hat[0:5]
