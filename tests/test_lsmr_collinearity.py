"""
Tests for multicollinearity detection with LSMR vs MAP backends.

When covariates are structurally nested within fixed effects, they are
collinear with the fixed effect dummies. Demeaning should project this out
entirely (zeroing the column), but LSMR's iterative solver may not demean
precisely enough for the Cholesky check alone to detect this at default
tolerances — the variance ratio check (`collin_tol_var`) adds an additional
check.

See: https://github.com/py-econometrics/pyfixest/issues/1042
     https://github.com/py-econometrics/pyfixest/issues/1139
"""

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feols


@pytest.fixture(scope="module")
def data_three_fe_nested():
    "Hard data set for LSMR."
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
    "Cholesky check alone does NOT catch the nested covariate with LSMR."
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="scipy",
        collin_tol_var=0,
    )
    assert "worker_educ" not in fit._collin_vars


def test_both_checks_catch_for_lsmr(data_three_fe_nested):
    "Cholesky + variance ratio check together catch the nested covariate."
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="scipy",
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_cholesky_alone_catches_for_map(data_three_fe_nested):
    "Cholesky check alone catches the nested covariate with MAP (numba)."
    fit = feols(
        FML,
        data=data_three_fe_nested,
        demeaner_backend="numba",
        collin_tol_var=0,
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames
