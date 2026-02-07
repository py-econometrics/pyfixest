"""
Tests for multicollinearity detection with LSMR vs MAP backends.

When covariates are structurally nested within fixed effects, they become
perfectly collinear after demeaning. LSMR's iterative solver may not demean
precisely enough for the Cholesky check alone to detect this — the variance
ratio check (`collin_tol_var`) fills that gap.

See: https://github.com/py-econometrics/pyfixest/issues/1042
     https://github.com/py-econometrics/pyfixest/issues/1139
"""

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols


@pytest.fixture(scope="module")
def data_three_fe_nested():
    """
    Three-way FE with unbalanced assignment — hard case for LSMR.

    Workers randomly assigned to firms across years. `worker_educ` is
    constant within worker -> collinear with worker FE. The random
    worker-firm-year assignment creates a poorly conditioned FE matrix
    where LSMR struggles to fully project out nested covariates.
    """
    rng = np.random.default_rng(42)
    n = 50_000

    worker_id = rng.integers(0, 5000, n)
    firm_id = rng.integers(0, 500, n)
    year_id = rng.integers(0, 10, n)

    educ_vals = rng.normal(0, 10_000, 5000)
    worker_educ = educ_vals[worker_id]

    indiv_x = rng.normal(0, 1, n)
    y = 2.0 * indiv_x + 0.5 * worker_educ + rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "Y": y,
            "indiv_x": indiv_x,
            "worker_educ": worker_educ,
            "worker_id": worker_id,
            "firm_id": firm_id,
            "year_id": year_id,
        }
    )


FML = "Y ~ indiv_x + worker_educ | worker_id + firm_id + year_id"


def test_cholesky_alone_misses_for_lsmr(data_three_fe_nested):
    """
    (a) Cholesky check alone does NOT catch the nested covariate with LSMR.

    With the variance ratio check disabled (`collin_tol_var=0`), scipy's LSMR
    doesn't demean precisely enough for the Cholesky decomposition to detect
    that `worker_educ` is absorbed by the worker fixed effect.
    """
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="scipy",
        collin_tol_var=0,
    )
    assert "worker_educ" not in fit._collin_vars


def test_both_checks_catch_for_lsmr(data_three_fe_nested):
    """
    (b) Cholesky + variance ratio check together catch the nested covariate.

    With `collin_tol_var` auto-enabled (default for LSMR backends), the
    variance ratio check detects that `worker_educ` is absorbed by the
    worker fixed effect, even when Cholesky alone would miss it.
    """
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="scipy",
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_cholesky_alone_catches_for_map(data_three_fe_nested):
    """
    (c) Cholesky check alone catches the nested covariate with MAP (numba).

    MAP's absolute element-wise convergence criterion demeans precisely
    enough that the Cholesky decomposition detects the collinearity
    without needing the variance ratio check.
    """
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="numba",
        collin_tol_var=0,
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames
