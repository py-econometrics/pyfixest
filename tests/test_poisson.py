import contextlib
import os

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# rpy2 imports
from rpy2.robjects.packages import importr

import pyfixest as pf
from pyfixest.estimation.estimation import fepois

pandas2ri.activate()

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
@pytest.mark.parametrize("fml", ["Y ~ X1", "Y ~ X1 | f1"])
def test_against_fixest(fml):
    data = pf.get_data(model="Fepois")
    iwls_tol = 1e-12

    # vcov = "hetero"
    vcov = "hetero"
    fit = pf.fepois(fml, data=data, vcov=vcov, iwls_tol=iwls_tol)
    fit_r = fixest.fepois(ro.Formula(fml), data=data, vcov=vcov, glm_tol=iwls_tol)

    np.testing.assert_allclose(
        fit_r.rx2("irls_weights").reshape(-1, 1), fit._weights, atol=1e-08, rtol=1e-07
    )
    np.testing.assert_allclose(
        fit_r.rx2("linear.predictors").reshape(-1, 1),
        fit._Xbeta,
        atol=1e-08,
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        fit_r.rx2("scores").reshape(-1, 1),
        fit._scores.reshape(-1, 1),
        atol=1e-08,
        rtol=1e-07,
    )

    np.testing.assert_allclose(
        fit_r.rx2("hessian"), fit._hessian, atol=1e-08, rtol=1e-07
    )

    np.testing.assert_allclose(
        fit_r.rx2("deviance"), fit.deviance, atol=1e-08, rtol=1e-07
    )
