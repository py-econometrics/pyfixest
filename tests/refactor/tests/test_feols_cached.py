"""
New FEOLS tests using cached R results.
This replaces test_single_fit_feols with a cached approach.
"""
import pytest
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

import pyfixest as pf
from pyfixest.utils.utils import ssc

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.feols.test_generator import generate_feols_test_cases
from refactor.config.feols.test_cases import TestSingleFitFeols
from refactor.r_cache.feols.r_test_runner import FeolsRTestRunner


class CachedRResults:
    """Manages cached R results for comparison."""

    def __init__(self, cache_dir: str = "data/cached_results"):
        # Make cache_dir relative to the refactor directory (parent of tests directory)
        refactor_dir = Path(__file__).parent.parent
        self.cache_dir = refactor_dir / cache_dir
        self.runner = FeolsRTestRunner(str(self.cache_dir))

    def get_cached_result(self, test_case: TestSingleFitFeols):
        """Get cached R result for a test case."""
        cache_path = self.runner._get_cache_path(test_case)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached result not found for {test_case.test_id}")

        with open(cache_path, 'r') as f:
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


def _convert_f3(data: pd.DataFrame, f3_type: str) -> pd.DataFrame:
    """Convert f3 column to specified type (from original test)."""
    if f3_type == "categorical":
        data["f3"] = pd.Categorical(data["f3"])
    elif f3_type == "int":
        data["f3"] = data["f3"].astype(float).astype(np.int32)
    elif f3_type == "str":
        data["f3"] = data["f3"].astype(str)
    elif f3_type == "object":
        data["f3"] = data["f3"].astype(object)
    elif f3_type == "float":
        data["f3"] = data["f3"].astype(float)
    return data


def prepare_python_data(test_case: TestSingleFitFeols) -> pd.DataFrame:
    """Prepare data for Python test (same as R preparation)."""
    data_params = test_case.get_data_params()
    f3_type = data_params.pop('f3_type', 'str')

    # Generate base data
    data = pf.get_data(**data_params)

    if test_case.dropna:
        data = data.dropna()

    # Handle categorical conversion
    data[data == "nan"] = np.nan

    # Apply f3 type conversion
    data = _convert_f3(data, f3_type)

    return data


# Generate all test cases
ALL_FEOLS_TEST_CASES = generate_feols_test_cases()


@pytest.mark.against_r_core
@pytest.mark.parametrize("test_case", ALL_FEOLS_TEST_CASES)
def test_feols_vs_cached_r(test_case: TestSingleFitFeols):
    """Test pyfixest FEOLS against cached R results."""
    # Get cached R results
    try:
        r_results = CACHED_R_RESULTS.get_cached_result(test_case)
    except FileNotFoundError:
        pytest.fail(f"Cached R result not found for {test_case.test_id}")
        #pytest.skip(f"Cached R result not found for {test_case.test_id}")

    # Check if R test had an error
    if 'error' in r_results:
        pytest.skip(f"R test failed: {r_results['error']}")

    # Prepare data (same as R test)
    data = prepare_python_data(test_case)

    # Run Python test
    estimation_params = test_case.get_estimation_params()
    ssc_params = estimation_params['ssc']
    ssc_ = ssc(adj=ssc_params['adj'], cluster_adj=ssc_params['cluster_adj'])

    py_fit = pf.feols(
        fml=test_case.formula,
        data=data,
        vcov=test_case.inference,
        weights=test_case.weights,
        ssc=ssc_,
        demeaner_backend=test_case.demeaner_backend,
    )

    # Extract Python results
    try:
        py_coef = py_fit.coef().xs("X1")
    except KeyError:
        # If no X1, take first coefficient
        py_coef = py_fit.coef().iloc[0]

    py_n_coefs = py_fit.coef().values.size

    try:
        py_se = py_fit.se().xs("X1")
        py_pval = py_fit.pvalue().xs("X1")
        py_tstat = py_fit.tstat().xs("X1")
        py_confint = py_fit.confint().xs("X1").values
    except KeyError:
        # If no X1, take first coefficient
        py_se = py_fit.se().iloc[0]
        py_pval = py_fit.pvalue().iloc[0]
        py_tstat = py_fit.tstat().iloc[0]
        py_confint = py_fit.confint().iloc[0].values

    py_vcov = py_fit._vcov[0, 0]
    py_nobs = py_fit._N
    py_dof_k = getattr(py_fit, '_dof_k', None)
    py_df_t = getattr(py_fit, '_df_t', None)

    # Compare results with tolerances from original test
    rtol = 1e-08
    atol = 1e-08

    # Main comparisons
    check_absolute_diff(py_nobs, r_results['r_nobs'], atol, f"py_nobs != r_nobs for {test_case.test_id}")
    check_absolute_diff(py_coef, r_results['r_coef'], atol, f"py_coef != r_coef for {test_case.test_id}")
    check_absolute_diff(py_n_coefs, r_results['r_n_coefs'], atol, f"py_n_coefs != r_n_coefs for {test_case.test_id}")
    check_absolute_diff(py_se, r_results['r_se'], atol, f"py_se != r_se for {test_case.test_id}")
    check_absolute_diff(py_pval, r_results['r_pval'], atol, f"py_pval != r_pval for {test_case.test_id}")
    check_absolute_diff(py_tstat, r_results['r_tstat'], atol, f"py_tstat != r_tstat for {test_case.test_id}")
    check_absolute_diff(py_confint, r_results['r_confint'], atol, f"py_confint != r_confint for {test_case.test_id}")
    check_absolute_diff(py_vcov, r_results['r_vcov'], atol, f"py_vcov != r_vcov for {test_case.test_id}")
    check_absolute_diff(py_dof_k, r_results['r_dof_k'], atol, f"py_dof_k != r_dof_k for {test_case.test_id}")
    check_absolute_diff(py_df_t, r_results['r_df_t'], atol, f"py_df_t != r_df_t for {test_case.test_id}")

    # Additional comparisons for IID case (if available)
    if (test_case.inference == "iid" and test_case.ssc_adj and test_case.ssc_cluster_adj
        and r_results.get('r_resid') is not None):

        py_resid = py_fit.resid()
        py_predict = py_fit.predict()

        check_absolute_diff(
            py_predict[0:5],
            r_results['r_predict'],
            1e-07,
            f"py_predict != r_predict for {test_case.test_id}"
        )
        check_absolute_diff(
            py_resid[0:5],
            r_results['r_resid'],
            1e-07,
            f"py_resid != r_resid for {test_case.test_id}"
        )


# Script to generate all cached R results
def generate_all_r_cache():
    """Generate all R results and cache them."""
    runner = FeolsRTestRunner()
    test_cases = generate_feols_test_cases()

    print(f"Generating R cache for {len(test_cases)} test cases...")
    results = runner.run_all_tests(test_cases, force_refresh=False)

    successful = len([r for r in results.values() if 'error' not in r])
    failed = len([r for r in results.values() if 'error' in r])

    print(f"Generated {successful} cached results")
    if failed > 0:
        print(f"Failed to generate {failed} results")

    return results


if __name__ == "__main__":
    # Generate cache when run as script
    generate_all_r_cache()
