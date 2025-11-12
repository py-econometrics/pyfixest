"""Test pyfixest against R fixest using mtcars dataset."""

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

import pyfixest as pf

pandas2ri.activate()

fixest = importr("fixest")
stats = importr("stats")
broom = importr("broom")

# Tolerance for comparing results
rtol = 1e-06
atol = 1e-06


def check_absolute_diff(x1, x2, tol, msg=None):
    """Check for absolute differences."""
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


def _get_r_tidy(r_fit):
    """Get tidied results from R fixest fit."""
    tidied_r = broom.tidy_fixest(r_fit, conf_int=ro.BoolVector([False]))
    df_r = pd.DataFrame(tidied_r).T
    df_r.columns = ["term", "estimate", "std.error", "statistic", "p.value"]
    return df_r.set_index("term")


@pytest.fixture(scope="module")
def mtcars_data():
    """Load mtcars dataset from R."""
    mtcars = ro.r["mtcars"]
    return pandas2ri.rpy2py(mtcars)


@pytest.mark.against_r_core
@pytest.mark.parametrize("formula", [
    "mpg ~ hp + wt + C(cyl)",
    "mpg ~ hp + wt + C(cyl) + C(gear)",
    "mpg ~ hp * wt + C(cyl)",
])
@pytest.mark.parametrize("vcov", ["iid", "hetero"])
def test_feols_mtcars(mtcars_data, formula, vcov):
    """Test feols against R fixest using mtcars."""
    # Convert formula for R
    r_formula = formula.replace("C(", "factor(")
    
    # Fit in R
    r_fit = fixest.feols(
        ro.Formula(r_formula),
        data=mtcars_data,
        vcov=vcov,
        ssc=fixest.ssc(True, "none", False, True, "min", "min"),
    )
    
    # Fit in Python
    py_fit = pf.feols(
        fml=formula,
        data=mtcars_data,
        vcov=vcov,
        ssc=pf.ssc(k_adj=True, k_fixef="none", G_adj=True),
    )
    
    # Get results
    r_tidy = _get_r_tidy(r_fit)
    py_tidy = py_fit.tidy()
    
    # Compare coefficient for 'wt'
    r_coef = r_tidy.loc["wt", "estimate"]
    py_coef = py_tidy.loc["wt", "Estimate"]
    check_absolute_diff(py_coef, r_coef, atol, f"Coefficients don't match for {formula}")
    
    # Compare standard error for 'wt'
    r_se = r_tidy.loc["wt", "std.error"]
    py_se = py_tidy.loc["wt", "Std. Error"]
    check_absolute_diff(py_se, r_se, atol, f"Standard errors don't match for {formula}, vcov={vcov}")


@pytest.mark.against_r_core
@pytest.mark.parametrize("formula", [
    "mpg ~ hp + wt + C(cyl)",
    "mpg ~ hp + wt + C(cyl) + C(gear)",
    "mpg ~ hp * wt + C(cyl)",
])
@pytest.mark.parametrize("vcov", ["iid", "hetero"])
def test_feglm_gaussian_mtcars(mtcars_data, formula, vcov):
    """Test feglm with Gaussian family against R fixest using mtcars."""
    # Convert formula for R
    r_formula = formula.replace("C(", "factor(")
    
    # Fit in R
    r_fit = fixest.feglm(
        ro.Formula(r_formula),
        data=mtcars_data,
        family=stats.gaussian(),
        vcov=vcov,
    )
    
    # Fit in Python
    py_fit = pf.feglm(
        fml=formula,
        data=mtcars_data,
        family="gaussian",
        vcov=vcov,
    )
    
    # Get results
    r_tidy = _get_r_tidy(r_fit)
    py_tidy = py_fit.tidy()
    
    # Compare coefficient for 'wt'
    r_coef = r_tidy.loc["wt", "estimate"]
    py_coef = py_tidy.loc["wt", "Estimate"]
    check_absolute_diff(py_coef, r_coef, atol, f"Coefficients don't match for {formula}")
    
    # Compare standard error for 'wt'
    r_se = r_tidy.loc["wt", "std.error"]
    py_se = py_tidy.loc["wt", "Std. Error"]
    check_absolute_diff(
        py_se, r_se, atol, 
        f"Standard errors don't match for feglm {formula}, vcov={vcov}"
    )


@pytest.mark.against_r_core
@pytest.mark.parametrize("formula", [
    "mpg ~ hp + wt | cyl",
    "mpg ~ hp + wt | cyl + gear",
])
@pytest.mark.parametrize("vcov", ["iid", "hetero"])
def test_feglm_gaussian_with_fe_mtcars(mtcars_data, formula, vcov):
    """Test feglm with Gaussian family and fixed effects against R fixest using mtcars."""
    # Fit in R
    r_fit = fixest.feglm(
        ro.Formula(formula),
        data=mtcars_data,
        family=stats.gaussian(),
        vcov=vcov,
    )
    
    # Fit in Python
    py_fit = pf.feglm(
        fml=formula,
        data=mtcars_data,
        family="gaussian",
        vcov=vcov,
    )
    
    # Get results
    r_tidy = _get_r_tidy(r_fit)
    py_tidy = py_fit.tidy()
    
    # Compare coefficient for 'wt'
    r_coef = r_tidy.loc["wt", "estimate"]
    py_coef = py_tidy.loc["wt", "Estimate"]
    check_absolute_diff(
        py_coef, r_coef, atol, 
        f"Coefficients don't match for {formula} with FE"
    )
    
    # Compare standard error for 'wt'
    r_se = r_tidy.loc["wt", "std.error"]
    py_se = py_tidy.loc["wt", "Std. Error"]
    check_absolute_diff(
        py_se, r_se, atol, 
        f"Standard errors don't match for feglm {formula} with FE, vcov={vcov}"
    )


