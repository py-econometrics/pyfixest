from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation import fepois

fixest = importr("fixest")


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
@pytest.mark.parametrize(
    "example,missing_rows",
    [
        ("reproduction", [0, 1, 2, 3, 4]),
        ("reproduction", [150]),
        ("reproduction", [0, 8, 32, 70, 121, 169, 230, 299]),
        ("ppmlhdfe", [4]),
    ],
)
def test_ir_separation_after_missing_rows(example, missing_rows):
    """IR separation preserves row identity after formula-level filtering."""
    if example == "ppmlhdfe":
        data = pd.read_csv(
            Path(__file__).parent / "data/ppmlhdfe_separation_examples/01.csv"
        )
        fml = "y ~ x1 + x2 | id1 + id2"
        regressor = "x1"
    else:
        rng = np.random.default_rng(3)
        n = 300
        data = pd.DataFrame({"f1": rng.integers(0, 10, n), "X1": rng.normal(size=n)})
        data["D"] = (rng.uniform(size=n) < 0.1).astype(float)
        data["Y"] = rng.poisson(np.exp(0.5 + 0.3 * data["X1"]))
        data.loc[data["D"] == 1, "Y"] = 0
        data["separated"] = data["D"]
        fml = "Y ~ X1 + D | f1"
        regressor = "X1"

    data.loc[missing_rows, regressor] = np.nan
    complete = data.dropna()
    retained = complete.loc[complete["separated"] == 0]
    n_separated = int(complete["separated"].sum())
    fits = []
    for sample in (data, complete.reset_index(drop=True)):
        with pytest.warns(
            UserWarning,
            match=rf"{n_separated} observations removed because of separation\.",
        ):
            fits.append(
                pf.fepois(fml, data=sample, separation_check=["ir"], iwls_tol=1e-12)
            )
    fit, prefiltered = fits
    assert fit.sample_info.n_rows == prefiltered.sample_info.n_rows == len(retained)
    assert fit.sample_info.dropped_by_stage.separation == n_separated
    assert prefiltered.sample_info.dropped_by_stage.separation == n_separated
    pd.testing.assert_index_equal(fit.model_matrix.dependent.index, retained.index)
    pd.testing.assert_series_equal(fit.coef(), prefiltered.coef())

    # fixest does not implement IR; compare its fit on the known nonseparated
    # sample (the ppmlhdfe fixture records the externally identified rows).
    fit_r = fixest.fepois(ro.Formula(fml), data=retained, glm_tol=1e-12)
    coef_r = ro.r["coef"](fit_r)
    coef_names = ro.r("function(fit) names(coef(fit))")(fit_r)
    expected = pd.Series(np.asarray(coef_r), index=list(coef_names))
    assert set(fit.coef().index) == set(expected.index)
    # Allow small differences from IRLS stopping and fixed-effect projection.
    np.testing.assert_allclose(
        fit.coef(),
        expected.loc[fit.coef().index],
        rtol=1e-7,
        atol=1e-7,
        err_msg="Poisson coefficients after IR separation differ from fixest",
    )


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
