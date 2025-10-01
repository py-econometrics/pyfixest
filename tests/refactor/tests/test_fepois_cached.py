"""
New FEPOIS tests using cached R results.
This replaces test_single_fit_fepois with a cached approach.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf

# Pre-import commonly used functions to avoid repeated lookups
from pyfixest import fepois
from pyfixest.utils.utils import ssc

# Add tests directory to path for imports
tests_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tests_dir))

from refactor.config.fepois.test_cases import TestSingleFitFepois
from refactor.config.fepois.test_generator import generate_fepois_test_cases
from refactor.r_cache.fepois.r_test_runner import FepoisRTestRunner


class CachedRResults:
    """Manages cached R results for comparison with preloaded cache."""

    def __init__(self, cache_dir: str = "data/cached_results"):
        # Make cache_dir relative to the refactor directory (parent of tests directory)
        refactor_dir = Path(__file__).parent.parent
        self.cache_dir = refactor_dir / cache_dir
        self.runner = FepoisRTestRunner(str(self.cache_dir))
        # Preload ALL cache at startup for performance
        self._memory_cache = self._preload_all_cache()

    def _preload_all_cache(self):
        """Preload all cached results into memory at startup."""
        print("Preloading all cached FEPOIS R results...")
        cache = {}

        if not self.cache_dir.exists():
            print(f"Cache directory {self.cache_dir} does not exist")
            return cache

        # Look for FEPOIS cache files in the method-specific subdirectory
        fepois_cache_dir = self.cache_dir / "fepois"
        if not fepois_cache_dir.exists():
            print(f"FEPOIS cache directory {fepois_cache_dir} does not exist")
            return cache

        # Load all JSON files at once
        json_files = list(fepois_cache_dir.glob("*.json"))
        print(f"Loading {len(json_files)} cached FEPOIS results...")

        for cache_file in json_files:
            try:
                with open(cache_file) as f:
                    result = json.load(f)
                # Extract test_id from filename (remove hash suffix)
                test_id = cache_file.stem.split('_')[0] + '_' + cache_file.stem.split('_')[1]
                cache[test_id] = result
            except Exception as e:
                print(f"Failed to load {cache_file}: {e}")

        print(f"Preloaded {len(cache)} cached FEPOIS results")
        return cache

    def get_cached_result(self, test_case: TestSingleFitFepois):
        """Get cached R result for a test case."""
        test_id = test_case.test_id

        if test_id not in self._memory_cache:
            raise FileNotFoundError(f"Cached result not found for {test_case.test_id}")

        return self._memory_cache[test_id]


# Global cached results manager
CACHED_R_RESULTS = CachedRResults()


def extract_python_results_optimized(py_mod):
    """Extract all Python results in one optimized pass."""
    # Get all DataFrames once
    coef_df = py_mod.coef()
    se_df = py_mod.se()
    pval_df = py_mod.pvalue()
    tstat_df = py_mod.tstat()
    confint_df = py_mod.confint()

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
        'vcov': py_mod._vcov[0, 0],
        'nobs': py_mod._N,
        'dof_k': getattr(py_mod, "_dof_k", None),
        'df_t': getattr(py_mod, "_df_t", None),
        'deviance': py_mod.deviance,
        'resid': py_mod.resid(),
        'irls_weights': py_mod._irls_weights.flatten(),
    }


def run_all_comparisons_vectorized(py_results, r_results, test_case, fepois_tol=1e-04, fepois_tol_crv=1e-03):
    """Run all comparisons with minimal function call overhead."""
    test_id = test_case.test_id

    # Use relaxed tolerance for CRV inference, standard for others
    tolerance = (
        fepois_tol_crv
        if isinstance(test_case.inference, dict) and "CRV1" in test_case.inference
        else fepois_tol
    )

    # Convert all R results to numpy arrays upfront
    r_arrays = {
        'nobs': np.array([r_results["nobs"]]),
        'coef': np.array([r_results["coef"]]),
        'n_coefs': np.array([r_results["n_coefs"]]),
        'se': np.array([r_results["se"]]),
        'pval': np.array([r_results["pval"]]),
        'tstat': np.array([r_results["tstat"]]),
        'confint_0': np.array([r_results["confint"][0]]),
        'confint_1': np.array([r_results["confint"][1]]),
        'vcov': np.array([r_results["vcov"]]),
        'dof_k': np.array([int(r_results["dof_k"])]),
        'df_t': np.array([int(r_results["df_t"])]),
        'deviance': np.array([r_results["deviance"]]),
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
        'deviance': np.array([py_results['deviance']]),
    }

    # Define tolerances for each comparison
    tolerances = {
        'nobs': fepois_tol,
        'coef': fepois_tol,
        'n_coefs': fepois_tol,
        'se': tolerance,
        'pval': tolerance,
        'tstat': tolerance,
        'confint_0': tolerance,
        'confint_1': tolerance,
        'vcov': tolerance,
        'dof_k': fepois_tol,
        'df_t': fepois_tol,
        'deviance': fepois_tol,
    }

    # Run all comparisons
    for key in r_arrays:
        py_val = py_arrays[key]
        r_val = r_arrays[key]
        tol = tolerances[key]

        # Check NaN positions
        nan_mask_py = np.isnan(py_val)
        nan_mask_r = np.isnan(r_val)

        if not np.array_equal(nan_mask_py, nan_mask_r):
            raise AssertionError(f"py_{key} != r_{key} for {test_id}: NaN positions do not match")

        # Check values
        valid_mask = ~nan_mask_py
        if not np.all(np.abs(py_val[valid_mask] - r_val[valid_mask]) < tol):
            raise AssertionError(f"py_{key} != r_{key} for {test_id}")


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


def _has_fixed_effects(formula: str) -> bool:
    """Check if formula contains fixed effects (indicated by | symbol)."""
    return "|" in formula


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


# Cache for prepared data to avoid repeated generation
_DATA_CACHE = {}

# Cache for SSC objects to avoid repeated creation
_SSC_CACHE = {}


def prepare_python_data(test_case: TestSingleFitFepois) -> pd.DataFrame:
    """Prepare data for Python FEPOIS test, matching R preparation."""
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

    # Apply dropna if needed (should always be False for FEPOIS)
    if test_case.dropna:
        data = data.dropna()

    # Handle categorical conversion (matching R script)
    data[data == "nan"] = np.nan

    # Apply f3 type conversion
    data = _convert_f3(data, f3_type)

    # Cache the result
    _DATA_CACHE[cache_key] = data.copy()

    return data


# Generate all test cases once at module level for better performance
ALL_FEPOIS_TEST_CASES = generate_fepois_test_cases()


@pytest.mark.parametrize("test_case", ALL_FEPOIS_TEST_CASES)
def test_fepois_vs_cached_r(test_case: TestSingleFitFepois):
    """Test FEPOIS Python implementation against cached R results."""
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

    # Cache SSC objects to avoid repeated creation
    ssc_key = (ssc_params["adj"], ssc_params["cluster_adj"])
    if ssc_key not in _SSC_CACHE:
        _SSC_CACHE[ssc_key] = ssc(adj=ssc_params["adj"], cluster_adj=ssc_params["cluster_adj"])
    ssc_ = _SSC_CACHE[ssc_key]

    # Run Python FEPOIS estimation
    py_mod = fepois(
        fml=test_case.formula,
        data=data,
        vcov=test_case.inference,
        ssc=ssc_,
        iwls_tol=test_case.iwls_tol,
        iwls_maxiter=test_case.iwls_maxiter,
    )

    # Extract Python results using optimized function
    py_results = extract_python_results_optimized(py_mod)

    # Get predictions (skip if formula has fixed effects - not supported for FEPOIS)
    py_predict_response = None
    py_predict_link = None
    if not _has_fixed_effects(test_case.formula):
        py_predict_response = py_mod.predict(type="response")
        py_predict_link = py_mod.predict(type="link")

    # Use vectorized comparison for main results
    try:
        run_all_comparisons_vectorized(py_results, r_results, test_case)
    except AssertionError as e:
        pytest.fail(str(e))

    # Residuals and IRLS weights (only for iid case)
    if test_case.inference == "iid" and test_case.ssc_adj and test_case.ssc_cluster_adj:
        check_absolute_diff(
            py_results['resid'][0:5], r_results["resid"][0:5], 1e-06, "py_resid != r_resid"
        )  # Relaxed for FEPOIS
        check_absolute_diff(
            py_results['irls_weights'][10:12],
            r_results["irls_weights"][10:12],
            1e-05,  # More relaxed tolerance for IRLS weights in FEPOIS
            "py_irls_weights != r_irls_weights",
        )

    # Prediction comparisons (only if predictions are available)
    if py_predict_response is not None and r_results.get("predict_response") is not None:
        check_absolute_diff(
            py_predict_response[0:5],
            r_results["predict_response"][0:5],
            1e-06,  # Relaxed tolerance for FEPOIS predictions
            "py_predict_response != r_predict_response",
        )
    if py_predict_link is not None and r_results.get("predict_link") is not None:
        check_absolute_diff(
            py_predict_link[0:5],
            r_results["predict_link"][0:5],
            1e-06,  # Relaxed tolerance for FEPOIS predictions
            "py_predict_link != r_predict_link",
        )


if __name__ == "__main__":
    # Allow running this file directly for debugging
    test_cases = generate_fepois_test_cases()
    print(f"Generated {len(test_cases)} FEPOIS test cases")

    # Run first test case as example
    if test_cases:
        print(f"Example test case: {test_cases[0].test_id}")
        print(f"Formula: {test_cases[0].formula}")
        print(f"Inference: {test_cases[0].inference}")
