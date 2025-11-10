"Pytest configuration for pyfixest tests."

import os
import sys

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
    arg in sys.argv
    for arg in ["test_hac_vs_fixest.py", "-m hac", "--markers=hac"]
) or any("test_hac_vs_fixest" in arg for arg in sys.argv)

if _run_hac_tests:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
