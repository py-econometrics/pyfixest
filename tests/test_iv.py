import numpy as np
import pytest

from pyfixest.estimation.estimation import feols
from pyfixest.utils.utils import get_data


@pytest.mark.parametrize("seed", [293, 912])
@pytest.mark.parametrize("sd", [0.1, 0.2, 0.3])
def test_1st_stage_iv(seed, sd):
    # Test 1st stage regression result in 2SLS estimator.
    rng = np.random.default_rng(seed)
    data = get_data().dropna()
    data["Z1"] = data["Z1"] + rng.normal(0, sd, size=len(data))

    # Compute test statistics of IV and OLS respectively

    fit_iv = feols("Y ~ 1 | f1 | X1 ~ Z1 ", data=data)

    _pi_hat_iv = fit_iv._pi_hat
    _X_hat_iv = fit_iv._X_hat
    _v_hat_iv = fit_iv._v_hat

    fit_ols = feols("X1 ~  Z1 | f1", data=data)

    _pi_hat_ols = fit_ols._beta_hat
    _X_hat_ols = fit_ols._Y_hat_link
    _v_hat_ols = fit_ols._u_hat

    # Assert that the parameter estimates and predicted values are c
    # lose between IV and OLS
    np.testing.assert_allclose(
        _pi_hat_iv,
        _pi_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="First stage coefficient estimate mismatch between IV and OLS",
    )

    np.testing.assert_allclose(
        _X_hat_iv,
        _X_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Predicted X values mismatch in first stage between IV and OLS",
    )

    np.testing.assert_allclose(
        _v_hat_iv,
        _v_hat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Residuals mismatch in first stage between IV and OLS",
    )
