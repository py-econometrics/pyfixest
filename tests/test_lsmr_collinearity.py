"""
Tests for multicollinearity detection with the LSMR solver (scipy backend).

When covariates are structurally nested within fixed effects, they become
perfectly collinear after demeaning. The LSMR solver must demean accurately
enough that the downstream collinearity check detects and drops
these variables — matching the behavior of the MAP (numba) backend.

See: https://github.com/py-econometrics/pyfixest/issues/1042
     https://github.com/py-econometrics/pyfixest/issues/1139
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from pyfixest.estimation import feglm, feols, fepois


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_nested():
    """
    Covariate nested within a single FE (easy case).

    `group_level_x` is constant within `group_id` → perfectly collinear
    with the group FE after demeaning. Large magnitude (std=100) stresses
    LSMR's relative stopping criterion.
    """
    rng = np.random.default_rng(42)
    n_groups = 50
    obs_per_group = 40
    n = n_groups * obs_per_group

    group_id = np.repeat(np.arange(n_groups), obs_per_group)
    group_level_x = np.repeat(rng.normal(0, 100, n_groups), obs_per_group)
    indiv_x = rng.normal(0, 1, n)
    y = 2.0 * indiv_x + 0.5 * group_level_x + rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "Y": y,
            "indiv_x": indiv_x,
            "group_level_x": group_level_x,
            "group_id": group_id,
        }
    )


@pytest.fixture(scope="module")
def data_two_fe_nested():
    """
    Covariate nested within one of two crossed FEs.

    Mimics the structure from issue #1042: `property_id` and `year^fire_id`,
    where a fire-level covariate is constant within property_id groups.
    """
    rng = np.random.default_rng(123)
    n_groups = 30
    n_periods = 10
    n = n_groups * n_periods

    group_id = np.repeat(np.arange(n_groups), n_periods)
    time_id = np.tile(np.arange(n_periods), n_groups)
    group_level_x = np.repeat(rng.normal(0, 500, n_groups), n_periods)
    indiv_x = rng.normal(0, 1, n)
    y = 3.0 * indiv_x + 1.0 * group_level_x + rng.normal(0, 0.5, n)

    return pd.DataFrame(
        {
            "Y": y,
            "indiv_x": indiv_x,
            "group_level_x": group_level_x,
            "group_id": group_id,
            "time_id": time_id,
        }
    )


@pytest.fixture(scope="module")
def data_categorical_nested():
    """
    Categorical dummies nested within a FE.

    C(region) is constant within unit_id → all dummies collinear with unit FE.
    """
    rng = np.random.default_rng(99)
    n_units = 40
    obs_per_unit = 15
    n = n_units * obs_per_unit

    unit_id = np.repeat(np.arange(n_units), obs_per_unit)
    region = np.repeat(rng.choice([0, 1, 2], n_units), obs_per_unit)
    x = rng.normal(0, 1, n)
    y = x + 0.5 * region + rng.normal(0, 1, n)

    return pd.DataFrame(
        {
            "Y": y,
            "x": x,
            "region": region,
            "unit_id": unit_id,
        }
    )


@pytest.fixture(scope="module")
def data_three_fe_nested():
    """
    Three-way FE with unbalanced assignment — hard case for LSMR.

    Workers randomly assigned to firms across years. `worker_educ` is
    constant within worker → collinear with worker FE. The random
    worker-firm-year assignment creates a poorly conditioned FE matrix
    where LSMR struggles to fully project out nested covariates.

    This is the closest to the real-world structure from issue #1042
    (property_id × year^fire_id with fire-level covariates).
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


@pytest.fixture(scope="module")
def data_nested_poisson():
    """
    Data for Poisson regression with nested covariate.

    `group_level_x` is constant within `group_id` → collinear with group FE.
    Y is Poisson-distributed count data.
    """
    rng = np.random.default_rng(42)
    n_groups = 20
    obs_per_group = 10
    n = n_groups * obs_per_group

    group_id = np.repeat(np.arange(n_groups), obs_per_group)
    group_level_x = np.repeat(rng.normal(0, 1, n_groups), obs_per_group)
    indiv_x = rng.normal(0, 0.5, n)
    linear_pred = 0.5 * indiv_x - 0.3 * group_level_x
    y = rng.poisson(np.exp(linear_pred))

    return pd.DataFrame(
        {
            "Y": y,
            "indiv_x": indiv_x,
            "group_level_x": group_level_x,
            "group_id": group_id,
        }
    )


# ---------------------------------------------------------------------------
# Easy cases: one or two FEs — scipy detects collinearity at all tolerances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixef_tol", [1e-4, 1e-6, 1e-8])
def test_scipy_drops_nested_covariate(data_nested, fixef_tol):
    """Covariate constant within group should be dropped with scipy backend."""
    fit = feols(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


@pytest.mark.parametrize("fixef_tol", [1e-4, 1e-6, 1e-8])
def test_scipy_drops_nested_two_fe(data_two_fe_nested, fixef_tol):
    """Covariate nested in one FE should be dropped with two-way FE."""
    fit = feols(
        "Y ~ indiv_x + group_level_x | group_id + time_id",
        data=data_two_fe_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


@pytest.mark.parametrize("fixef_tol", [1e-4, 1e-6, 1e-8])
def test_scipy_drops_nested_categorical(data_categorical_nested, fixef_tol):
    """Categorical dummies nested within FE should be dropped."""
    fit = feols(
        "Y ~ x + C(region) | unit_id",
        data=data_categorical_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    region_collin = [v for v in fit._collin_vars if "region" in v]
    assert len(region_collin) > 0, (
        f"Expected region dummies dropped, got {fit._collin_vars}"
    )
    assert "x" in fit._coefnames


# ---------------------------------------------------------------------------
# Hard case: three-way FE — variance ratio check catches what Cholesky misses
# ---------------------------------------------------------------------------


def test_variance_check_catches_three_fe_at_default_tol(data_three_fe_nested):
    """
    At default fixef_tol=1e-6, scipy now drops the nested covariate
    thanks to the variance ratio check (auto-enabled for LSMR backends).

    Without the variance ratio check, the Cholesky check alone misses it
    because LSMR doesn't demean precisely enough at default tolerance.
    """
    fit = feols(
        "Y ~ indiv_x + worker_educ | worker_id + firm_id + year_id",
        data=data_three_fe_nested,
        demeaner_backend="scipy",
        fixef_tol=1e-6,
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_variance_check_disabled_misses_three_fe(data_three_fe_nested):
    """
    With collin_tol_var=0 (variance check disabled), scipy misses
    the nested covariate at default tolerance — proving the variance
    ratio check is what catches it.
    """
    fit = feols(
        "Y ~ indiv_x + worker_educ | worker_id + firm_id + year_id",
        data=data_three_fe_nested,
        demeaner_backend="scipy",
        fixef_tol=1e-6,
        collin_tol_var=0,
    )
    assert "worker_educ" not in fit._collin_vars


def test_scipy_catches_nested_three_fe_at_tight_tol(data_three_fe_nested):
    """
    At fixef_tol=1e-8, scipy succeeds even without variance ratio check —
    the tighter tolerance gives LSMR enough precision for the Cholesky check.
    """
    fit = feols(
        "Y ~ indiv_x + worker_educ | worker_id + firm_id + year_id",
        data=data_three_fe_nested,
        demeaner_backend="scipy",
        fixef_tol=1e-8,
        collin_tol_var=0,  # disable variance check
    )
    assert "worker_educ" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_numba_always_catches_nested_three_fe(data_three_fe_nested):
    """numba (MAP) catches nested collinearity at all tolerances."""
    for tol in [1e-2, 1e-4, 1e-6, 1e-8]:
        fit = feols(
            "Y ~ indiv_x + worker_educ | worker_id + firm_id + year_id",
            data=data_three_fe_nested,
            demeaner_backend="numba",
            fixef_tol=tol,
        )
        assert "worker_educ" in fit._collin_vars, (
            f"numba failed to drop worker_educ at fixef_tol={tol}"
        )


# ---------------------------------------------------------------------------
# Variance ratio check: no false positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixef_tol", [1e-4, 1e-6, 1e-8])
def test_variance_check_keeps_valid_covariates(data_nested, fixef_tol):
    """Non-collinear covariates should NOT be dropped at any tolerance."""
    fit = feols(
        "Y ~ indiv_x | group_id",
        data=data_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    assert fit._collin_vars == []
    assert "indiv_x" in fit._coefnames


def test_variance_check_with_numba_opt_in(data_nested):
    """Variance check can be opted-in for numba backend."""
    fit = feols(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        demeaner_backend="numba",
        collin_tol_var=1e-6,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


# ---------------------------------------------------------------------------
# scipy matches numba on easy cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixef_tol", [1e-6, 1e-8])
def test_scipy_matches_numba_nested(data_nested, fixef_tol):
    """Scipy and numba backends should drop the same collinear variables."""
    fit_numba = feols(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        demeaner_backend="numba",
        fixef_tol=fixef_tol,
    )
    fit_scipy = feols(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    assert set(fit_numba._collin_vars) == set(fit_scipy._collin_vars)
    assert set(fit_numba._coefnames) == set(fit_scipy._coefnames)
    np.testing.assert_allclose(
        fit_numba.coef().values, fit_scipy.coef().values, atol=1e-4
    )


@pytest.mark.parametrize("fixef_tol", [1e-6, 1e-8])
def test_scipy_matches_numba_two_fe(data_two_fe_nested, fixef_tol):
    """Scipy and numba should agree on collinearity with two-way FE."""
    fit_numba = feols(
        "Y ~ indiv_x + group_level_x | group_id + time_id",
        data=data_two_fe_nested,
        demeaner_backend="numba",
        fixef_tol=fixef_tol,
    )
    fit_scipy = feols(
        "Y ~ indiv_x + group_level_x | group_id + time_id",
        data=data_two_fe_nested,
        demeaner_backend="scipy",
        fixef_tol=fixef_tol,
    )
    assert set(fit_numba._collin_vars) == set(fit_scipy._collin_vars)
    assert set(fit_numba._coefnames) == set(fit_scipy._coefnames)
    np.testing.assert_allclose(
        fit_numba.coef().values, fit_scipy.coef().values, atol=1e-4
    )


# ---------------------------------------------------------------------------
# fepois: variance ratio check
# ---------------------------------------------------------------------------


def test_fepois_drops_nested_covariate(data_nested_poisson):
    """fepois should drop covariate nested within FE."""
    fit = fepois(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested_poisson,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_fepois_variance_check_catches_nested(data_nested_poisson):
    """fepois with collin_tol_var opt-in drops nested covariate."""
    fit = fepois(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested_poisson,
        collin_tol_var=1e-6,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


# ---------------------------------------------------------------------------
# feglm: variance ratio check
# ---------------------------------------------------------------------------


def test_feglm_drops_nested_covariate(data_nested):
    """feglm (gaussian) should drop covariate nested within FE."""
    fit = feglm(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        family="gaussian",
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames


def test_feglm_variance_check_catches_nested(data_nested):
    """feglm (gaussian) with collin_tol_var opt-in drops nested covariate."""
    fit = feglm(
        "Y ~ indiv_x + group_level_x | group_id",
        data=data_nested,
        family="gaussian",
        collin_tol_var=1e-6,
    )
    assert "group_level_x" in fit._collin_vars
    assert "indiv_x" in fit._coefnames
