import re

import numpy as np

import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data


def test_gelbach_example():
    test_ci = False

    data = gelbach_data(nobs = 10_000)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    res = fit.decompose(
        param="x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}
    ).GelbachDecompositionResults

    np.testing.assert_allclose(res.contribution_dict.get("g1"), np.array([2.468092]))
    np.testing.assert_allclose(res.contribution_dict.get("g2"), np.array([1.068156]))

    np.testing.assert_allclose(
        res.contribution_dict.get("direct_effect"), np.array([4.608666])
    )
    np.testing.assert_allclose(
        res.contribution_dict.get("full_effect"), np.array([1.072417])
    )
    np.testing.assert_allclose(
        res.contribution_dict.get("explained_effect"), np.array([3.536249])
    )

    if test_ci:
        np.testing.assert_allclose(res.ci.get("g1"), np.array([2.293714, 2.64247]))
        np.testing.assert_allclose(res.ci.get("g2"), np.array([2.9546626, 1.18165]))

        np.testing.assert_allclose(
            res.ci.get("direct_effect"), np.array([4.401328, 4.816004])
        )
        np.testing.assert_allclose(
            res.ci.get("full_effect"), np.array([0.9775936, 1.167241])
        )
        np.testing.assert_allclose(
            res.ci.get("explained_effect"), np.array([3.3262, 3.746298])
        )


def test_regex():
    "Test the regex functionality for combine_covariates."
    data = gelbach_data(nobs = 100)
    fit1 = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    fit2 = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    fit1.decompose(
        param="x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}, seed = 3, reps = 100
    )

    fit2.decompose(
        param = "x1", combine_covariates = {"g1": re.compile(r"x2[1-2]"), "g2": ["x23"]}, seed = 3, reps = 100
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )