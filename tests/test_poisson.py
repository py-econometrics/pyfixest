from __future__ import annotations

import contextlib
import os

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation import fepois
from pyfixest.estimation.formula.parse import Formula
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
            demeaner=MapDemeaner(),
        )
        == expected
    )


def test_separation_ir_requires_rectifier_updates():
    """Multiple rectifier iterations recover the ppmlhdfe separation mask."""
    data = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "data/ppmlhdfe_separation_examples/15.csv"
        )
    )
    arguments = {
        "Y": data[["y"]],
        "X": data[["x1", "x2", "x3"]],
        # One constant FE represents the intercept in the reference dataset.
        "fe": pd.DataFrame({"intercept": np.zeros(len(data), dtype=np.int64)}),
        "demeaner": MapDemeaner(),
    }
    with pytest.warns(RuntimeWarning, match="maximum number of iterations"):
        assert _check_for_separation_ir(**arguments, maxiter=1) == set()
    expected = set(data.index[data.separated == 1])
    assert expected
    assert _check_for_separation_ir(**arguments) == expected


def test_separation():
    """Test separation detection."""
    example1 = pd.DataFrame.from_dict(
        {
            "Y": [0, 0, 0, 1, 2, 3],
            "fe1": ["a", "a", "b", "b", "b", "c"],
            "fe2": ["c", "c", "d", "d", "d", "e"],
            "X": np.random.normal(0, 1, 6),
        }
    )
    with pytest.warns(
        UserWarning, match="2 observations removed because of separation."
    ):
        fepois("Y ~ X  | fe1", data=example1, vcov="hetero", separation_check=["fe"])

    if False:
        # this example is taken from ppmlhdfe's primer on separation https://github.com/sergiocorreia/ppmlhdfe/blob/master/guides/separation_primer.md
        # disabled because we currently do not perform separation checks if no fixed effects are provided
        # TODO: enable once separation checks without fixed effects are enabled
        example2 = pd.DataFrame.from_dict(
            {
                "Y": [0, 0, 0, 1, 2, 3],
                "X1": [2, -1, 0, 0, 5, 6],
                "X2": [5, 10, 0, 0, -10, -12],
            }
        )

        with pytest.warns(
            UserWarning, match="2 observations removed because of separation."
        ):
            fepois("Y ~ X1 + X2", data=example2, vcov="hetero", separation_check=["ir"])

    # ppmlhdfe test data sets (check readme in data/ppmlhdfe_separation_examples)
    path = os.path.dirname(os.path.abspath(__file__))
    folder = r"data/ppmlhdfe_separation_examples"
    fns = sorted(
        [fn for fn in os.listdir(os.path.join(path, folder)) if fn.endswith(".csv")]
    )
    for fn in fns:
        if fn in ["07.csv"]:
            # this case fails but is not tested in ppmlhdfe
            # https://github.com/sergiocorreia/ppmlhdfe/blob/master/test/validate_tagsep.do#L27
            continue
        data = pd.read_csv(os.path.join(path, folder, fn))
        # build formula dynamically from dataframe
        # datasets have fixed structure of the form (y, x1, ..., xN, id1, ..., idM, separated)
        fml = "y"  # dependent variable y
        regressors = data.columns[
            data.columns.str.startswith("x")
        ]  # regressors x1,...,xN
        fixed_effects = data.columns[
            data.columns.str.startswith("id")
        ]  # fixed effects id1,...,id2

        if regressors.empty:
            # TODO: formulae with just a constant term and fixed effects throw error in FIT.get_fit(), e.g., for 03.csv and Y ~ 1 | id1 + id2 + id3?
            continue
        fml += f" ~ {' + '.join(regressors)}"

        if fixed_effects.empty:
            # TODO: separation checks are currently disabled if no fixed effects are specified; enable tests once we run separation check without fixed effects
            continue
        else:
            fml += f" | {' + '.join(fixed_effects)}"

        with (
            pytest.warns(
                UserWarning,
                match=f"{data.separated.sum()} observations removed because of separation.",
            ) as record,
            contextlib.suppress(Exception),
        ):
            pf.fepois(fml, data=data, separation_check=["ir"])

        # if no separation, no warning is raised
        if data.separated.sum() == 0:
            assert len(record) == 0


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
def test_separation_preserves_response_and_auxiliary_columns(lhs, monkeypatch):
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
    parse = Formula.parse
    calls = []

    def parse_once(formula):
        calls.append(formula)
        assert len(calls) == 1, "Separation must not parse an auxiliary string."
        return parse(formula)

    monkeypatch.setattr(Formula, "parse", parse_once)
    fit = pf.fepois(f"{lhs} ~ X - 1 | fe", data=data, separation_check=["ir"])
    assert fit.sample_info.n_obs == expected.sample_info.n_obs == 6
    # Scaling a Poisson response shifts the FE intercepts, leaving its slope.
    np.testing.assert_allclose(fit.coef(), expected.coef(), rtol=1e-6, atol=1e-8)
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("regressor", ["x", "np.sin(center(x))"])
def test_separation_preserves_fitted_transform_state(regressor):
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
    reference_regressor = "x" if regressor == "x" else "transformed_x"
    expected = pf.fepois(
        f"y ~ {reference_regressor} | g",
        data=data,
        fixef_rm="none",
        separation_check=["ir"],
    )
    fit = pf.fepois(
        f"y ~ {regressor} | np.floor(center(c))",
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
        fit.predict(),
        expected.predict(),
        rtol=1e-8,
        atol=1e-10,
        err_msg="IR changed retained-row predictions",
    )


@pytest.mark.parametrize(
    "name", ["U", "omega", "U_separationTmp", "omega_separationTmp"]
)
def test_separation_preserves_context_predictors(name):
    """IR does not shadow predictors supplied through the formula context."""
    rng = np.random.default_rng(415)
    data = pd.DataFrame(
        {"y": rng.poisson(1, 100), "x": rng.normal(size=100), "f": np.arange(100) % 5}
    )
    expected = pf.fepois("y ~ x | f", data=data, separation_check=["ir"])
    fit = pf.fepois(
        f"y ~ {name} | f",
        data=data,
        context={name: data.x},
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
