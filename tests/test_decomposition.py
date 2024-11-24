import re

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data


@pytest.mark.parametrize("only_coef", [True, False])
def test_gelbach_example(only_coef):
    data = gelbach_data(nobs=10_000)
    data["f"] = np.random.choice(10, size=data.shape[0])

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    fit.decompose(
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        reps=100,
        seed=8,
        only_coef=only_coef,
    )

    res = fit.GelbachDecompositionResults

    np.testing.assert_allclose(
        res.contribution_dict.get("g1"), np.array([2.468092]), atol=1e-1
    )
    np.testing.assert_allclose(
        res.contribution_dict.get("g2"), np.array([1.068156]), atol=1e-1
    )

    np.testing.assert_allclose(
        res.contribution_dict.get("direct_effect"), np.array([4.608666]), atol=1e-1
    )
    np.testing.assert_allclose(
        res.contribution_dict.get("full_effect"), np.array([1.072417]), atol=1e-1
    )
    np.testing.assert_allclose(
        res.contribution_dict.get("explained_effect"), np.array([3.536249]), atol=1e-1
    )
    if False:
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

    if False:
        # clustered errors:
        fit.vcov({"CRV1": "f"})
        fit.decompose(
            param="x1",
            combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
            reps=100,
            seed=8,
        )

        # no combine covariates
        fit.decompose(param="x1", reps=100, seed=8)


def test_regex():
    "Test the regex functionality for combine_covariates."
    data = gelbach_data(nobs=100)
    fit1 = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    fit2 = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    fit1.decompose(
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        seed=3,
        reps=100,
    )

    fit2.decompose(
        param="x1",
        combine_covariates={"g1": re.compile(r"x2[1-2]"), "g2": ["x23"]},
        seed=3,
        reps=100,
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )


def test_agg_first():
    "Test that choosing agg_first = True or False does not change the results."
    data = pf.get_data(N=400)
    fit1 = pf.feols("Y ~ X1 + C(f1) + C(f2)", data=data)
    fit2 = pf.feols("Y ~ X1 + C(f1) + C(f2)", data=data)

    fit1.decompose(
        param="X1",
        combine_covariates={
            "f1": re.compile(r"\b\w*f1\w*\b"),
            "f2": re.compile(r"\b\w*f2\w*\b"),
        },
        nthreads=2,
        agg_first=False,
        reps=3,
        seed=123,
    )

    fit2.decompose(
        param="X1",
        combine_covariates={
            "f1": re.compile(r"\b\w*f1\w*\b"),
            "f2": re.compile(r"\b\w*f2\w*\b"),
        },
        nthreads=2,
        agg_first=True,
        reps=3,
        seed=123,
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )
