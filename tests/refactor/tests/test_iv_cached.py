"""
New IV tests using cached R results.
This replaces test_single_fit_iv with a cached approach.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.utils.utils import ssc

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.iv.test_cases import TestSingleFitIv
from refactor.config.iv.test_generator import generate_iv_test_cases
from refactor.r_cache.iv.r_test_runner import IvRTestRunner


class CachedRResults:
    """Manages cached R results for comparison."""

    def __init__(self, cache_dir: str = "data/cached_results"):
        # Make cache_dir relative to the refactor directory (parent of tests directory)
        refactor_dir = Path(__file__).parent.parent
        self.cache_dir = refactor_dir / cache_dir
        self.runner = IvRTestRunner(str(self.cache_dir))

    def get_cached_result(self, test_case: TestSingleFitIv):
        """Get cached R result for a test case."""
        cache_path = self.runner._get_cache_path(test_case)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached R result not found for {test_case.test_id}"
            )

        with open(cache_path) as f:
            return json.load(f)


# Global cached results manager
CACHED_R_RESULTS = CachedRResults()


def check_absolute_diff(x1, x2, tol, msg=None):
    """Check for absolute differences (from original test)."""
    # Handle None values (from R's NULL)
    if x1 is None and x2 is None:
        return  # Both None, considered equal
    if x1 is None or x2 is None:
        raise AssertionError(f"{msg}: One value is None, the other is not")

    # Convert to numpy arrays
    if isinstance(x1, (int, float)):
        x1 = np.array([x1])
    elif isinstance(x1, list):
        x1 = np.array(x1)

    if isinstance(x2, (int, float)):
        x2 = np.array([x2])
    elif isinstance(x2, list):
        x2 = np.array(x2)

    msg = "" if msg is None else msg

    # Handle nan values
    nan_mask_x1 = np.isnan(x1)
    nan_mask_x2 = np.isnan(x2)

    if not np.array_equal(nan_mask_x1, nan_mask_x2):
        raise AssertionError(f"{msg}: NaN positions do not match")

    valid_mask = ~nan_mask_x1
    assert np.all(np.abs(x1[valid_mask] - x2[valid_mask]) < tol), msg


def _convert_f3(data, f3_type):
    """Convert f3 column to specified type (matching FEOLS implementation)."""
    if f3_type == "str":
        data["f3"] = data["f3"].astype(str)
    elif f3_type == "object":
        data["f3"] = data["f3"].astype(object)
    elif f3_type == "int":
        data["f3"] = data["f3"].astype(int)
    elif f3_type == "categorical":
        data["f3"] = data["f3"].astype("category")
    elif f3_type == "float":
        data["f3"] = data["f3"].astype(float)
    return data


def prepare_python_data(test_case: TestSingleFitIv) -> pd.DataFrame:
    """Prepare data for Python IV test, matching R preparation."""
    data_params = test_case.get_data_params()
    f3_type = data_params.pop("f3_type", "str")

    # Generate base data
    data = pf.get_data(**data_params)

    # Apply dropna if needed (should always be False for IV)
    if test_case.dropna:
        data = data.dropna()

    # Handle categorical conversion (matching R script)
    data[data == "nan"] = np.nan

    # Apply f3 type conversion
    data = _convert_f3(data, f3_type)

    return data


@pytest.mark.parametrize("test_case", generate_iv_test_cases())
def test_iv_vs_cached_r(test_case: TestSingleFitIv):
    """Test IV Python implementation against cached R results."""
    # Get cached R results
    try:
        r_results = CACHED_R_RESULTS.get_cached_result(test_case)
    except FileNotFoundError:
        pytest.fail(f"Cached R result not found for {test_case.test_id}")

    # Check if R test was successful
    if not r_results.get("success", False):
        pytest.fail(
            f"R test failed for {test_case.test_id}: {r_results.get('error', 'Unknown error')}"
        )

    # Prepare Python data
    data = prepare_python_data(test_case)

    # Get estimation parameters
    estimation_params = test_case.get_estimation_params()
    ssc_params = estimation_params["ssc"]
    ssc_ = ssc(adj=ssc_params["adj"], cluster_adj=ssc_params["cluster_adj"])

    # Run Python IV estimation (IV uses feols, not a separate IV function)
    py_mod = pf.feols(
        fml=test_case.formula,
        data=data,
        vcov=test_case.inference,
        ssc=ssc_,
        weights=test_case.weights,
    )

    # Extract Python results
    py_coef = py_mod.coef().xs("X1")
    py_se = py_mod.se().xs("X1")
    py_pval = py_mod.pvalue().xs("X1")
    py_tstat = py_mod.tstat().xs("X1")
    py_confint = py_mod.confint().xs("X1").values
    py_nobs = py_mod._N
    py_vcov = py_mod._vcov[0, 0]
    py_dof_k = getattr(py_mod, '_dof_k', None)
    py_df_t = getattr(py_mod, '_df_t', None)
    py_n_coefs = py_mod.coef().values.size

    # Get residuals (predictions not supported for IV models in pyfixest)
    py_predict = None  # IV predictions not supported in pyfixest
    py_resid = py_mod.resid()

    # Extract R results
    r_coef = r_results["coef"]
    r_se = r_results["se"]
    r_pval = r_results["pval"]
    r_tstat = r_results["tstat"]
    r_confint = r_results["confint"]
    r_nobs = r_results["nobs"]
    r_vcov = r_results["vcov"]
    r_dof_k = r_results["dof_k"]
    r_df_t = r_results["df_t"]
    r_n_coefs = r_results["n_coefs"]
    r_predict = r_results["predict"]
    r_resid = r_results["resid"]

    # Perform comparisons (based on original test conditions)
    # Only test when inference is "iid" and both adj and cluster_adj are True
    if test_case.inference == "iid" and test_case.ssc_adj and test_case.ssc_cluster_adj:
        check_absolute_diff(py_nobs, r_nobs, 1e-08, "py_nobs != r_nobs")
        check_absolute_diff(py_coef, r_coef, 1e-08, "py_coef != r_coef")
        check_absolute_diff(py_se, r_se, 1e-08, "py_se != r_se")
        check_absolute_diff(py_tstat, r_tstat, 1e-08, "py_tstat != r_tstat")
        check_absolute_diff(py_pval, r_pval, 1e-08, "py_pval != r_pval")
        check_absolute_diff(py_confint, r_confint, 1e-08, "py_confint != r_confint")
        check_absolute_diff(py_vcov, r_vcov, 1e-08, "py_vcov != r_vcov")
        check_absolute_diff(py_dof_k, r_dof_k, 1e-08, "py_dof_k != r_dof_k")
        check_absolute_diff(py_df_t, r_df_t, 1e-08, "py_df_t != r_df_t")
        check_absolute_diff(py_n_coefs, r_n_coefs, 1e-08, "py_n_coefs != r_n_coefs")

        # Residuals and predictions comparisons (only if available)
        if r_resid is not None:
            check_absolute_diff(
                py_resid[0:5], r_resid[0:5], 1e-07, "py_resid != r_resid"
            )

        # Skip prediction comparisons for IV (not supported in pyfixest)
        # if py_predict is not None and r_predict is not None:
        #     check_absolute_diff(
        #         py_predict[0:5], r_predict[0:5], 1e-07, "py_predict != r_predict"
        #     )


if __name__ == "__main__":
    # Allow running this file directly for debugging
    test_cases = generate_iv_test_cases()
    print(f"Generated {len(test_cases)} IV test cases")

    # Run first test case as example
    if test_cases:
        print(f"Example test case: {test_cases[0].test_id}")
        print(f"Formula: {test_cases[0].formula}")
        print(f"Inference: {test_cases[0].inference}")
        print(f"Weights: {test_cases[0].weights}")
