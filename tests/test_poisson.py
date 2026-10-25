from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation import fepois
from pyfixest.estimation.internals.separation import _check_for_separation_ir

fixest = importr("fixest")


@pytest.mark.parametrize(
    "response, groups, expected",
    [
        pytest.param([1, 2, 3, 4], [0, 0, 1, 1], set(), id="positive-response"),
        pytest.param([1, 2, 0, 0, 0], [0, 0, 0, 1, 1], {3, 4}, id="fixed-effects-only"),
    ],
)
def test_separation_ir_without_regressors(response, groups, expected):
    """Only all-zero FE groups are separated when there are no regressors."""
    dependent = pd.DataFrame({"y": response})
    assert (
        _check_for_separation_ir(
            Y=dependent,
            X=pd.DataFrame(index=dependent.index),
            fe=pd.DataFrame({"group": groups}),
            outcome_bounds=(0.0,),
            demeaner=MapDemeaner(),
            collin_tol=1e-9,
        )
        == expected
    )


@pytest.mark.parametrize(
    "dataset_path",
    sorted((Path(__file__).parent / "data/ppmlhdfe_separation_examples").glob("*.csv")),
    ids=lambda path: path.name,
)
def test_separation_ir_datasets(dataset_path):
    """IR recovers the exact separation masks from the ppmlhdfe fixtures."""
    if dataset_path.name == "07.csv":
        # Also excluded by ppmlhdfe's test/validate_tagsep.do.
        pytest.skip("07.csv is excluded from the upstream separation validation.")

    data = pd.read_csv(dataset_path)
    regressors = data.filter(regex="^x")
    if regressors.empty:
        # Preserve the existing dataset test's FE-only exclusions.
        pytest.skip("IR does not recover the multiway FE-only reference masks.")
    fixed_effects = data.filter(regex="^id")
    if fixed_effects.empty:
        # One constant FE represents the intercept in the reference datasets.
        fixed_effects = pd.DataFrame({"intercept": np.zeros(len(data), dtype=np.int64)})
    else:
        fixed_effects = fixed_effects.apply(lambda column: pd.factorize(column)[0])

    arguments = {
        "Y": data[["y"]],
        "X": regressors,
        "fe": fixed_effects,
        "outcome_bounds": (0.0,),
        "demeaner": MapDemeaner(),
        "collin_tol": 1e-9,
    }
    expected = set(data.index[data.separated == 1])
    if dataset_path.name == "15.csv":
        assert expected
        with pytest.warns(RuntimeWarning, match="maximum number of iterations"):
            assert _check_for_separation_ir(**arguments, maxiter=1) == set()
    assert _check_for_separation_ir(**arguments) == expected


def test_separation_fe():
    """The public Poisson fit removes observations in all-zero FE levels."""
    rng = np.random.default_rng(0)
    example1 = pd.DataFrame.from_dict(
        {
            "Y": [0, 0, 0, 1, 2, 3],
            "fe1": ["a", "a", "b", "b", "b", "c"],
            "fe2": ["c", "c", "d", "d", "d", "e"],
            "X": rng.normal(0, 1, 6),
        }
    )
    with pytest.warns(
        UserWarning, match="2 observations removed because of separation."
    ):
        fepois("Y ~ X  | fe1", data=example1, vcov="hetero", separation_check=["fe"])


@pytest.mark.against_r_core
@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
def test_against_fixest(fml):
    data = pf.get_data(model="Fepois")
    iwls_tol = 1e-12

    # vcov = "hetero"
    vcov = "hetero"
    fit = pf.fepois(fml, data=data, vcov=vcov, iwls_tol=iwls_tol)
    fit_r = fixest.fepois(ro.Formula(fml), data=data, vcov=vcov, glm_tol=iwls_tol)

    np.testing.assert_allclose(
        fit_r.rx2("irls_weights"),
        fit.working_state.working_weights,
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("linear.predictors").reshape(-1, 1),
        fit.working_state.eta.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("scores").reshape(-1, 1),
        fit.sandwich.scores.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )

    np.testing.assert_allclose(
        fit_r.rx2("hessian"), fit.sandwich.hessian, atol=1e-08, rtol=1e-07
    )

    np.testing.assert_allclose(
        fit_r.rx2("deviance"), fit.fitstat.deviance, atol=1e-08, rtol=1e-07
    )


@pytest.mark.parametrize("lhs", ["`my outcome`", "{Y * 2}"])
def test_separation_preserves_response_and_auxiliary_columns(lhs):
    data = pd.DataFrame(
        {
            "Y": [0, 0, 0, 1, 2, 3, 1, 2],
            "X": [1, 2, 1, 2, 3, 4, 2, 3],
            "fe": ["a", "a", "b", "b", "b", "b", "b", "b"],
            "U": np.arange(8),
            "U_separationTmp": np.arange(8),
            "omega": np.ones(8),
            "Uhat": np.ones(8),
        }
    )
    data["my outcome"] = data.Y
    original = data.copy(deep=True)
    expected = pf.fepois("Y ~ X - 1 | fe", data=data, separation_check=["ir"])
    fit = pf.fepois(f"{lhs} ~ X - 1 | fe", data=data, separation_check=["ir"])
    retained_rows = [2, 3, 4, 5, 6, 7]
    assert fit.model_matrix.dependent.index.tolist() == retained_rows
    assert expected.model_matrix.dependent.index.tolist() == retained_rows
    # Scaling a Poisson response shifts the FE intercepts, leaving its slope.
    np.testing.assert_allclose(fit.coef(), expected.coef(), rtol=1e-6, atol=1e-8)
    pd.testing.assert_frame_equal(data, original)


def test_separation_preserves_fitted_transform_state():
    """IR uses the original transform state despite missing-response rows."""
    rng = np.random.default_rng(4)
    data = pd.DataFrame(
        {
            "c": np.repeat([-0.4, 0.4, 1.3, 3.8], [40, 10, 5, 10]),
            "y": np.repeat([1.0, 0.0, 1.0, np.nan], [40, 10, 5, 10]),
            "x": rng.normal(size=65),
        }
    )
    data["g"] = np.floor(data.c - data.c.mean())
    data["transformed_x"] = np.sin(data.x - data.x.mean())
    expected = pf.fepois(
        "y ~ transformed_x | g",
        data=data,
        fixef_rm="none",
        separation_check=["ir"],
    )
    fit = pf.fepois(
        "y ~ np.sin(center(x)) | g",
        data=data,
        fixef_rm="none",
        separation_check=["ir"],
    )
    assert fit.sample_info.n_obs == expected.sample_info.n_obs == 55
    np.testing.assert_allclose(
        fit.coef(),
        expected.coef(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed the design learned before missing-response filtering",
    )
    np.testing.assert_allclose(
        fit.se(),
        expected.se(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed standard errors for the fitted transform state",
    )
    np.testing.assert_allclose(
        fit.predict(),
        expected.predict(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed retained-row predictions",
    )


def test_separation_preserves_context_predictor():
    """A context predictor gives the same fit as the corresponding data column."""
    rng = np.random.default_rng(415)
    data = pd.DataFrame(
        {"y": rng.poisson(1, 100), "x": rng.normal(size=100), "f": np.arange(100) % 5}
    )
    expected = pf.fepois("y ~ x | f", data=data, separation_check=["ir"])
    fit = pf.fepois(
        "y ~ omega | f",
        data=data,
        context={"omega": data.x},
        separation_check=["ir"],
    )
    assert fit.sample_info.n_obs == expected.sample_info.n_obs == 100
    np.testing.assert_allclose(
        fit.coef(),
        expected.coef(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed a context-supplied predictor",
    )
    np.testing.assert_allclose(
        fit.se(),
        expected.se(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed standard errors for a context-supplied predictor",
    )
    rows = [0, 33, 66, 99]
    np.testing.assert_allclose(
        fit.predict()[rows],
        expected.predict()[rows],
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed predictions for a context-supplied predictor",
    )
