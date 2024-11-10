import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from pyfixest.estimation.estimation import feols
from pyfixest.utils.set_rpy2_path import update_r_paths
from pyfixest.utils.utils import get_data

update_r_paths()

# Activate pandas2ri to enable conversion between pandas DataFrames and R DataFrames
pandas2ri.activate()

# Import the ivDiag package
ivDiag = importr("ivDiag")


# Enable the automatic conversion between pandas DataFrame and R DataFrame
pandas2ri.activate()


@pytest.mark.extended
@pytest.fixture(scope="module")
def r_results():
    np.random.seed(1)

    # Number of observations
    n = 500

    # Simulate the data
    # Instrumental variable
    z = np.random.binomial(1, 0.5, size=n)
    z2 = np.random.binomial(1, 0.5, size=n)
    # Endogenous variable
    d = 0.5 * z + 1.5 * z2 + np.random.normal(size=n)

    # Control variables
    c1 = np.random.normal(size=n)
    c2 = np.random.normal(size=n)

    # Outcome variable
    y = 1.0 + 1.5 * d + 0.8 * c1 + 0.5 * c2 + np.random.normal(size=n)

    # Cluster variable
    cluster = np.random.randint(1, 50, size=n)

    # Sampling weights (random uniform distribution between 1 and 3 for example)
    weights = np.random.uniform(1, 3, size=n)

    # Create a DataFrame
    data = pd.DataFrame(
        {
            "d": d,
            "y": y,
            "z": z,
            "z2": z2,
            "c1": c1,
            "c2": c2,
            "cluster": cluster,
            "weights": weights,
        }
    )

    # Convert the DataFrame to an R DataFrame
    data_r = pandas2ri.py2rpy(data)

    # Define the variables
    Y = "y"
    D = "d"  # Endogenous treatment
    Z = "z"  # Instrumental variable
    controls = ["c1", "c2"]  # Covariates of control variables
    cl = "cluster"

    # Convert the variables and controls to R objects
    Y_r = ro.StrVector([Y])
    D_r = ro.StrVector([D])
    Z_r = ro.StrVector([Z])
    controls_r = ro.StrVector(controls)
    cl_r = ro.StrVector([cl])

    # Call the ivDiag function from the ivDiag package
    result_with_weights = ivDiag.ivDiag(
        Y=Y_r,
        D=D_r,
        Z=Z_r,
        controls=controls_r,
        data=data_r,
        weights="weights",
        cl=cl_r,
        run_AR=False,
        parallel=False,
        bootstrap=False,
    )

    result_without_weights = ivDiag.ivDiag(
        Y=Y_r,
        D=D_r,
        Z=Z_r,
        controls=controls_r,
        data=data_r,
        cl=cl_r,
        run_AR=False,
        parallel=False,
        bootstrap=False,
    )

    return {
        "with_weights": result_with_weights.rx2("F_stat"),
        "without_weights": result_without_weights.rx2("F_stat"),
        "data": data,
    }


@pytest.mark.extended
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("adj_vcov", ["iid", "hetero", {"CRV1": "cluster"}])
def test_iv_Fstat_ivDiag(has_weight, adj_vcov, r_results):
    # Compare weak iv test result(naive, robust, and clustered F stats )
    # with ivDiag package.
    # Set random seed for reproducibility

    data = r_results["data"]
    if has_weight:
        weight_detail_py = "weights"
        result = r_results["with_weights"]
    else:
        weight_detail_py = None
        result = r_results["without_weights"]
    """
    if adj_vcov == 0.0:
        vcov_detail = {"CRV1": "cluster"}
    elif adj_vcov == 1.0:
        vcov_detail = "iid"
    elif adj_vcov == 2.0:
        vcov_detail = "hetero"
    """
    fit_iv = feols(
        "y ~ 1 + c1 + c2 | d ~ z", data=data, vcov=adj_vcov, weights=weight_detail_py
    )
    fit_iv.first_stage()
    F_stat_pf = fit_iv._f_stat_1st_stage
    fit_iv.IV_Diag()
    F_stat_eff_pf = fit_iv._eff_F

    F_naive = result[0]
    F_hetero = result[1]
    F_cl = result[2]

    # Note that we are not putting arbitrary values into
    # F_eff_R. This is for saving computing times
    # Note that Effective F stat is equal to cluster robust F
    # when clusteres are set up. If not set up,
    # then effective F is equal to hetero-roboust F.
    _N = fit_iv._N
    if adj_vcov == {"CRV1": "cluster"}:
        F_stat_R = F_cl
        F_eff_R = result[3]
    elif adj_vcov == "iid":
        F_stat_R = F_naive
        F_eff_R = result[1]  # * _N / (_N - 1)
    elif adj_vcov == "hetero":
        F_stat_R = F_hetero  # * _N / (_N - 1)
        F_eff_R = result[1]  # * _N / (_N - 1)

    np.testing.assert_allclose(
        F_stat_pf,
        F_stat_R,
        rtol=1e-5,
        atol=1e-5,
        err_msg="First stage F stats estimate mismatch between pyfixest and IV_Diag packages",
    )
    np.testing.assert_allclose(
        F_stat_eff_pf,
        F_eff_R,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Effective F stats estimate mismatch between pyfixest and IV_Diag packages",
    )


@pytest.mark.parametrize("seed", [293, 912])
@pytest.mark.parametrize("sd", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("has_weight", [0.0, 1.0])
@pytest.mark.parametrize("adj_vcov", [0.0, 1.0, 2.0, 3.0, 4.0])
def test_1st_stage_iv(seed, sd, has_weight, adj_vcov):
    # Test 1st stage regression result in 2SLS estimator.
    rng = np.random.default_rng(seed)
    data = get_data().dropna()
    data["Z1"] = data["Z1"] + rng.normal(0, sd, size=len(data))

    # Compute test statistics of IV and OLS respectively

    weight_detail = "weights" if has_weight == 1.0 else None

    if adj_vcov == 0.0:
        vcov_detail = {"CRV1": "f1"}
    elif adj_vcov == 1.0:
        vcov_detail = "iid"
    elif adj_vcov == 2.0:
        vcov_detail = "hetero"
    elif adj_vcov == 3.0:
        vcov_detail = "HC1"
    elif adj_vcov == 4.0:
        vcov_detail = None

    fit_iv = feols(
        "Y ~ 1 | f1 | X1 ~ Z1 ", vcov=vcov_detail, data=data, weights=weight_detail
    )
    fit_ols = feols("X1 ~  Z1 | f1", vcov=vcov_detail, data=data, weights=weight_detail)

    fit_iv.first_stage()
    fit_ols.wald_test()

    _pi_hat_iv = fit_iv._pi_hat
    _X_hat_iv = fit_iv._X_hat
    _v_hat_iv = fit_iv._v_hat
    _F_stat_iv = fit_iv._f_stat_1st_stage
    _F_pval_iv = fit_iv._p_value_1st_stage

    _pi_hat_ols = fit_ols._beta_hat
    _X_hat_ols = fit_ols._X @ fit_ols._beta_hat
    _v_hat_ols = fit_ols._u_hat
    _F_stat_ols = fit_ols._f_statistic
    _F_pval_ols = fit_ols._p_value

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

    np.testing.assert_allclose(
        _F_stat_iv,
        _F_stat_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="F-Stats mismatch in first stage between IV and OLS",
    )

    np.testing.assert_allclose(
        _F_pval_iv,
        _F_pval_ols,
        rtol=1e-5,
        atol=1e-8,
        err_msg="F-Stats p-value mismatch in first stage between IV and OLS",
    )
