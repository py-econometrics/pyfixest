import re

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data


@pytest.fixture
def stata_results():
    # Define the data
    data = {
        "Coefficient": [
            2.432692,
            1.006828,
            3.43952,
            2.432692,
            1.006828,
            3.43952,
            1.919585,
            1.519934,
            3.43952,
            1.919585,
            1.519934,
            3.43952,
        ],
        "CI Lower": [
            2.190102,
            0.860606,
            3.142896,
            2.248167,
            0.833734,
            3.236171,
            1.7081,
            1.338977,
            3.142896,
            1.771888,
            1.356208,
            3.236171,
        ],
        "CI Upper": [
            2.675282,
            1.153049,
            3.736144,
            2.617217,
            1.179921,
            3.642868,
            2.13107,
            1.700891,
            3.736144,
            2.067283,
            1.68366,
            3.642868,
        ],
        "model": [
            "model 1",
            "model 1",
            "model 1",
            "model 1",
            "model 1",
            "model 1",
            "model 2",
            "model 2",
            "model 2",
            "model 2",
            "model 2",
            "model 2",
        ],
        "se": [
            "hetero",
            "hetero",
            "hetero",
            "cluster",
            "cluster",
            "cluster",
            "hetero",
            "hetero",
            "hetero",
            "cluster",
            "cluster",
            "cluster",
        ],
    }

    # Define the index
    coef_names = [
        "g1",
        "g2",
        "explained_effect",
        "g1",
        "g2",
        "explained_effect",
        "g1",
        "g2",
        "explained_effect",
        "g1",
        "g2",
        "explained_effect",
    ]

    # Create the DataFrame
    df = pd.DataFrame(data, index=coef_names)

    return df


def test_against_stata(stata_results):
    data = pd.read_stata("tests/data/gelbach.dta")
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    def decompose_and_compare(
        fit,
        stata_results,
        param,
        combine_covariates,
        seed,
        reps,
        model,
        se,
        cluster=None,
        agg_first=True,
    ):
        fit.decompose(
            param=param,
            combine_covariates=combine_covariates,
            seed=seed,
            reps=reps,
            cluster=cluster,
            agg_first=agg_first,
        )
        results = fit.GelbachDecompositionResults
        contribution_dict = results.contribution_dict
        ci = results.ci
        filtered_df = stata_results.query(f"model == '{model}' and se == '{se}'")

        for g in ["g1", "g2", "explained_effect"]:
            coef_diff = filtered_df.xs(g).Coefficient - contribution_dict[g]
            lower_diff = filtered_df.xs(g)["CI Lower"] - ci[g][0]
            upper_diff = filtered_df.xs(g)["CI Upper"] - ci[g][1]

            assert np.all(
                np.abs(coef_diff) < 1e-6
            ), f"Failed for {g} with values {filtered_df.xs(g).Coefficient} and {contribution_dict[g]}"
            if False:
                assert np.all(
                    np.abs(lower_diff) < 1e-4
                ), f"Failed for {g} with values {filtered_df.xs(g)['CI Lower']} and {ci[g][0]}"
                assert np.all(
                    np.abs(upper_diff) < 1e-4
                ), f"Failed for {g} with values {filtered_df.xs(g)['CI Upper']} and {ci[g][1]}"

    # Agg 1: Heteroskedastic SE
    decompose_and_compare(
        fit=fit,
        stata_results=stata_results,
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        seed=3,
        reps=100,
        model="model 1",
        se="hetero",
    )

    # Agg 1: Clustered SE
    decompose_and_compare(
        fit=fit,
        stata_results=stata_results,
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        seed=3,
        reps=100,
        model="model 1",
        se="cluster",
        cluster="cluster",
        agg_first=False,
    )

    # Agg 2: Heteroskedastic SE
    decompose_and_compare(
        fit=fit,
        stata_results=stata_results,
        param="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23"]},
        seed=3,
        reps=100,
        model="model 2",
        se="hetero",
    )

    # Agg 2: Clustered SE
    decompose_and_compare(
        fit=fit,
        stata_results=stata_results,
        param="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23"]},
        seed=3,
        reps=100,
        model="model 2",
        se="cluster",
        cluster="cluster",
        agg_first=False,
    )


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


def test_cluster():
    "Test that clustering is picked up correctly when set in feols, but not in decompose."
    df = pd.read_stata("tests/data/gelbach.dta")

    fit1 = pf.feols("y ~ x1 + x21 + x22 + x23", data=df, vcov={"CRV1": "cluster"})
    fit2 = pf.feols("y ~ x1 + x21 + x22 + x23", data=df)

    # cluster set in feols call
    fit1.decompose(
        param="x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}, digits=6
    )
    # cluster set in decompose
    fit2.decompose(
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        digits=6,
        cluster="cluster",
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )


def test_fixef():
    "Test that choosing agg_first = True or False does not change the results."
    df = pd.read_stata("tests/data/gelbach.dta")

    fit1 = pf.feols(
        "y ~ x1 + x21 + x22 + x23 | cluster", data=df, vcov={"CRV1": "cluster"}
    )
    fit2 = pf.feols("y ~ x1 + x21 + x22 + x23 + C(cluster)", data=df)

    fit1.decompose(
        param="x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}, digits=6
    )
    fit2.decompose(
        param="x1", combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]}, digits=6
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )


@pytest.mark.parametrize("agg_first", [True, False])
def test_combine_covariates_vs_none(agg_first):
    df = pd.read_stata("tests/data/gelbach.dta")
    fit1 = pf.feols("y ~ x1 + x21 + x22 + x23", data=df)
    fit2 = pf.feols("y ~ x1 + x21 + x22 + x23", data=df)

    fit1.decompose(param="x1", seed=3, reps=10, agg_first=agg_first)
    fit2.decompose(
        param="x1",
        combine_covariates={"x21": ["x21"], "x22": ["x22"], "x23": ["x23"]},
        seed=3,
        reps=10,
        agg_first=agg_first,
    )

    for key, value in fit1.GelbachDecompositionResults.contribution_dict.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.contribution_dict.get(key)
        )
