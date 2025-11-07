"Pytest configuration for pyfixest tests."

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def single_thread_blas():
    """
    Force single-threaded BLAS for deterministic HAC standard errors.

    What Claude says:

    Multi-threaded BLAS libraries (OpenBLAS, MKL, Accelerate) can produce
    slightly different numerical results due to different parallel reduction
    orders when computing matrix multiplications. This causes sporadic test
    failures in HAC variance calculations even though both R and Python
    implementations are mathematically correct.

    The differences arise because floating-point arithmetic is not associative:
    (a + b) + c ≠ a + (b + c) in IEEE 754. Different thread scheduling can
    change the order of operations, leading to different rounding errors.

    By forcing single-threaded execution, we ensure deterministic results
    that match R's fixest package exactly.
    """
    # Store original values to restore after tests
    original_values = {}

    env_vars = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)
        os.environ[var] = "1"

    yield

    # Restore original values after all tests complete
    for var, value in original_values.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
