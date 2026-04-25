"Pytest configuration for pyfixest tests."

import os
import sys

# Skip test files that require rpy2 when it is not installed (e.g. non-R pixi env).
_rpy2_test_files = [
    "test_did.py",
    "test_hac_vs_fixest.py",
    "test_i.py",
    "test_iv.py",
    "test_multcomp.py",
    "test_poisson.py",
    "test_predict_resid_fixef.py",
    "test_quantreg.py",
    "test_vs_fixest.py",
    "test_wald_test.py",
]
try:
    import pytest
    import rpy2  # noqa: F401
    import rpy2.robjects as ro
    from rpy2.rinterface import ListSexpVector
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import Converter
    from rpy2.robjects.vectors import ListVector

    def _build_rpy2_test_converter():
        """Build an rpy2 converter for tests compatible with fixest >= 0.14.

        We need pandas2ri's rpy2py conversion for numeric vectors so R
        predictions/residuals become numpy arrays in assertions. But the
        default list conversion path turns classed R lists into NamedList,
        dropping their R classes (e.g. `ssc.type`, `fixest`) and breaking
        round-trips when those objects are passed back to R.

        Override just ListSexpVector conversion so list-like R objects stay
        as rpy2 ListVector with class metadata preserved.
        """
        list_guard = Converter("pandas2ri-list-guard")

        @list_guard.rpy2py.register(ListSexpVector)
        def _keep_r_list_classes(obj):
            return ListVector(obj)

        return (
            ro.default_converter + numpy2ri.converter + pandas2ri.converter + list_guard
        )

    @pytest.fixture(scope="session", autouse=True)
    def _activate_pandas2ri():
        """Activate pandas->R conversion for the full test session (rpy2 >=3.6)."""
        with _build_rpy2_test_converter().context():
            yield

except ImportError:
    collect_ignore = [*_rpy2_test_files]

# Force single-threaded BLAS for deterministic HAC standard errors.
#
# Multi-threaded BLAS libraries (OpenBLAS, MKL, Accelerate) can produce
# slightly different numerical results due to different parallel reduction
# orders when computing matrix multiplications. This causes sporadic test
# failures in HAC variance calculations even though both R and Python
# implementations are mathematically correct.
#
# The differences arise because floating-point arithmetic is not associative:
# (a + b) + c ≠ a + (b + c) in IEEE 754. Different thread scheduling can
# change the order of operations, leading to different rounding errors.
#
# By forcing single-threaded execution, we ensure deterministic results
# that match R's fixest package exactly.
#
# This must be set at module level (before numpy/scipy imports) because
# BLAS libraries read these environment variables when first initialized.

# Check if HAC tests are being run
_run_hac_tests = any(
    arg in sys.argv for arg in ["test_hac_vs_fixest.py", "-m hac", "--markers=hac"]
) or any("test_hac_vs_fixest" in arg for arg in sys.argv)

if _run_hac_tests:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Trigger the maturin import hook build in the main process before xdist
# spawns workers. This avoids lock contention when multiple workers try to
# compile the Rust extension simultaneously.
import pyfixest  # noqa: F401, E402
