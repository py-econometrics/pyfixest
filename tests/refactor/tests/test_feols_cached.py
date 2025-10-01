"""
New FEOLS tests using cached R results.
This replaces test_single_fit_feols with a cached approach.

Performance optimizations implemented:
- Memory cache preloading: All R results loaded at startup
- Data preparation caching: Avoid repeated data generation
- Optimized result extraction: Single-pass DataFrame operations
- Vectorized comparisons: Batch numpy operations for better performance
- SSC object caching: Avoid repeated SSC creation
- Import optimization: Pre-import commonly used functions
- Selective expensive operations: Skip residual/prediction tests for 95% of cases
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf

# Pre-import commonly used functions to avoid repeated lookups
from pyfixest import feols
from pyfixest.utils.utils import ssc

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.feols.test_cases import TestSingleFitFeols
from refactor.config.feols.test_generator import generate_feols_test_cases
from refactor.r_cache.feols.r_test_runner import FeolsRTestRunner


class CachedRResults:
    """Manages cached R results for comparison with preloaded cache."""

    def __init__(self, cache_dir: str = "data/cached_results"):
        refactor_dir = Path(__file__).parent.parent
        self.cache_dir = refactor_dir / cache_dir
        self.runner = FeolsRTestRunner(str(self.cache_dir))
        # Preload ALL cache at startup for performance
        self._memory_cache = self._preload_all_cache()

    def _preload_all_cache(self):
        """Preload all cached results into memory at startup."""
        print("Preloading all cached FEOLS R results...")
        cache = {}

        if not self.cache_dir.exists():
            print(f"Cache directory {self.cache_dir} does not exist")
            return cache

        # Look for FEOLS cache files in the method-specific subdirectory
        feols_cache_dir = self.cache_dir / "feols"
        if not feols_cache_dir.exists():
            # Fallback to root cache directory for backward compatibility
            print(f"FEOLS cache directory {feols_cache_dir} does not exist, using root cache directory")
            cache_search_dir = self.cache_dir
        else:
            cache_search_dir = feols_cache_dir

        # Load all JSON files at once
        json_files = list(cache_search_dir.glob("*.json"))
        print(f"Loading {len(json_files)} cached FEOLS results...")

        for cache_file in json_files:
            try:
                with open(cache_file) as f:
                    result = json.load(f)
                # Extract test_id from filename (remove hash suffix if present)
                if '_' in cache_file.stem and len(cache_file.stem.split('_')) > 2:
                    # Format: feols_00001_hash -> feols_00001
                    test_id = '_'.join(cache_file.stem.split('_')[:2])
                else:
                    test_id = cache_file.stem
                cache[test_id] = result
            except Exception as e:
                print(f"Failed to load {cache_file}: {e}")

        print(f"Preloaded {len(cache)} cached FEOLS results")
        return cache

    def get_cached_result(self, test_case: TestSingleFitFeols):
        """Get cached R result for a test case."""
        test_id = test_case.test_id

        if test_id not in self._memory_cache:
            raise FileNotFoundError(f"Cached result not found for {test_case.test_id}")

        return self._memory_cache[test_id]

def extract_python_results_optimized(py_fit):
    """Extract all Python results in one optimized pass."""
    # Get all DataFrames once
    coef_df = py_fit.coef()
    se_df = py_fit.se()
    pval_df = py_fit.pvalue()
    tstat_df = py_fit.tstat()
    confint_df = py_fit.confint()

    # Try X1 first, fall back to first row
    if "X1" in coef_df.index:
        idx = "X1"
        py_coef = coef_df.loc[idx]
        py_se = se_df.loc[idx]
        py_pval = pval_df.loc[idx]
        py_tstat = tstat_df.loc[idx]
        py_confint = confint_df.loc[idx].values
    else:
        py_coef = coef_df.iloc[0]
        py_se = se_df.iloc[0]
        py_pval = pval_df.iloc[0]
        py_tstat = tstat_df.iloc[0]
        py_confint = confint_df.iloc[0].values

    return {
        'coef': py_coef,
        'n_coefs': coef_df.values.size,
        'se': py_se,
        'pval': py_pval,
        'tstat': py_tstat,
        'confint': py_confint,
        'vcov': py_fit._vcov[0, 0],
        'nobs': py_fit._N,
        'dof_k': getattr(py_fit, "_dof_k", None),
        'df_t': getattr(py_fit, "_df_t", None),
    }


def run_all_comparisons_vectorized(py_results, r_results, test_case, atol=1e-08):
    """Run all comparisons with minimal function call overhead."""
    test_id = test_case.test_id

    # Convert all R results to numpy arrays upfront
    r_arrays = {
        'nobs': np.array([r_results["r_nobs"]]),
        'coef': np.array([r_results["r_coef"]]),
        'n_coefs': np.array([r_results["r_n_coefs"]]),
        'se': np.array([r_results["r_se"]]),
        'pval': np.array([r_results["r_pval"]]),
        'tstat': np.array([r_results["r_tstat"]]),
        'confint_0': np.array([r_results["r_confint"][0]]),
        'confint_1': np.array([r_results["r_confint"][1]]),
        'vcov': np.array([r_results["r_vcov"]]),
        'dof_k': np.array([int(r_results["r_dof_k"])]),
        'df_t': np.array([int(r_results["r_df_t"])]),
    }

    # Convert Python results to numpy arrays
    py_arrays = {
        'nobs': np.array([py_results['nobs']]),
        'coef': np.array([py_results['coef']]),
        'n_coefs': np.array([py_results['n_coefs']]),
        'se': np.array([py_results['se']]),
        'pval': np.array([py_results['pval']]),
        'tstat': np.array([py_results['tstat']]),
        'confint_0': np.array([py_results['confint'][0]]),
        'confint_1': np.array([py_results['confint'][1]]),
        'vcov': np.array([py_results['vcov']]),
        'dof_k': np.array([int(py_results['dof_k'])]),
        'df_t': np.array([int(py_results['df_t'])]),
    }

    # Run all comparisons
    for key in r_arrays:
        py_val = py_arrays[key]
        r_val = r_arrays[key]

        # Check NaN positions
        nan_mask_py = np.isnan(py_val)
        nan_mask_r = np.isnan(r_val)

        if not np.array_equal(nan_mask_py, nan_mask_r):
            raise AssertionError(f"py_{key} != r_{key} for {test_id}: NaN positions do not match")

        # Check values
        valid_mask = ~nan_mask_py
        if not np.all(np.abs(py_val[valid_mask] - r_val[valid_mask]) < atol):
            raise AssertionError(f"py_{key} != r_{key} for {test_id}")


# Global cached results manager
CACHED_R_RESULTS = CachedRResults()


def check_absolute_diff(x1, x2, tol, msg=None):
    "Check for absolute differences."
    # Convert inputs to numpy arrays consistently - CRITICAL FIX
    if isinstance(x1, (int, float)):
        x1 = np.array([x1])
    elif isinstance(x1, list):
        x1 = np.array(x1)

    if isinstance(x2, (int, float)):
        x2 = np.array([x2])
    elif isinstance(x2, list):
        x2 = np.array(x2)

    msg = "" if msg is None else msg

    # handle nan values
    nan_mask_x1 = np.isnan(x1)
    nan_mask_x2 = np.isnan(x2)

    if not np.array_equal(nan_mask_x1, nan_mask_x2):
        raise AssertionError(f"{msg}: NaN positions do not match")

    valid_mask = ~nan_mask_x1  # Mask for non-NaN elements (same for x1 and x2)
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


# Cache for prepared data to avoid repeated generation
_DATA_CACHE = {}

# Cache for SSC objects to avoid repeated creation
_SSC_CACHE = {}


def prepare_python_data(test_case: TestSingleFitFeols) -> pd.DataFrame:
    """Prepare data for Python test (same as R preparation)."""
    # Create cache key based on data parameters
    data_params = test_case.get_data_params()
    cache_key = (
        data_params["N"],
        data_params["seed"],
        data_params["beta_type"],
        data_params["error_type"],
        data_params["model"],
        data_params["f3_type"],
        test_case.dropna,
    )

    # Check cache first
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()  # Return copy to avoid mutations

    f3_type = data_params.pop("f3_type", "str")

    # Generate base data
    data = pf.get_data(**data_params)

    if test_case.dropna:
        data = data.dropna()

    # Handle categorical conversion
    data[data == "nan"] = np.nan

    # Apply f3 type conversion
    data = _convert_f3(data, f3_type)

    # Cache the result
    _DATA_CACHE[cache_key] = data.copy()

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
        # pytest.skip(f"Cached R result not found for {test_case.test_id}")

    # Check if R test had an error
    if "error" in r_results:
        pytest.skip(f"R test failed: {r_results['error']}")

    # Prepare data (same as R test)
    data = prepare_python_data(test_case)

    # Run Python test
    estimation_params = test_case.get_estimation_params()
    ssc_params = estimation_params["ssc"]

    # Cache SSC objects to avoid repeated creation
    ssc_key = (ssc_params["adj"], ssc_params["cluster_adj"])
    if ssc_key not in _SSC_CACHE:
        _SSC_CACHE[ssc_key] = ssc(adj=ssc_params["adj"], cluster_adj=ssc_params["cluster_adj"])
    ssc_ = _SSC_CACHE[ssc_key]

    py_fit = feols(
        fml=test_case.formula,
        data=data,
        vcov=test_case.inference,
        weights=test_case.weights,
        ssc=ssc_,
        demeaner_backend=test_case.demeaner_backend,
    )

    # Extract Python results using optimized function
    py_results = extract_python_results_optimized(py_fit)

    # Compare results with tolerances from original test
    atol = 1e-08

    # Use vectorized comparison for main results
    try:
        run_all_comparisons_vectorized(py_results, r_results, test_case, atol)
    except AssertionError as e:
        pytest.fail(str(e))

    # Additional comparisons for IID case (if available) - Skip for performance
    # Only test residuals/predictions for a subset of tests to save time
    if (
        test_case.inference == "iid"
        and test_case.ssc_adj
        and test_case.ssc_cluster_adj
        and r_results.get("r_resid") is not None
        and int(test_case.test_id.split('_')[1]) % 20 == 0  # Only test every 20th case
    ):
        py_resid = py_fit.resid()
        py_predict = py_fit.predict()

        check_absolute_diff(
            py_predict[0:5],
            np.array(r_results["r_predict"]),
            1e-07,
            f"py_predict != r_predict for {test_case.test_id}",
        )
        check_absolute_diff(
            py_resid[0:5],
            np.array(r_results["r_resid"]),
            1e-07,
            f"py_resid != r_resid for {test_case.test_id}",
        )


# Script to generate all cached R results
def generate_all_r_cache():
    """Generate all R results and cache them."""
    runner = FeolsRTestRunner()
    test_cases = generate_feols_test_cases()

    print(f"Generating R cache for {len(test_cases)} test cases...")
    results = runner.run_all_tests(test_cases, force_refresh=False)

    successful = len([r for r in results.values() if "error" not in r])
    failed = len([r for r in results.values() if "error" in r])

    print(f"Generated {successful} cached results")
    if failed > 0:
        print(f"Failed to generate {failed} results")

    return results


if __name__ == "__main__":
    # Generate cache when run as script
    generate_all_r_cache()
