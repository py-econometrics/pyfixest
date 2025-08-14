import re

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.utils.dgps import gelbach_data

# Set matplotlib backend for headless testing
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.fixture
def gelbach_decomposition():
    """Fixture providing a standard Gelbach decomposition for testing."""
    data = gelbach_data(nobs=200)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    gb = fit.decompose(param="x1", seed=98765, reps=25)
    return gb


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


@pytest.mark.parametrize(
    "combine_covariates",
    [{"g1": ["x21", "x22"], "g2": ["x23"]}, {"g1": ["x21"], "g2": ["x22", "x23"]}],
)
@pytest.mark.parametrize("se", ["hetero", "cluster"])
@pytest.mark.parametrize("agg_first", [True, False])
def test_against_stata(stata_results, combine_covariates, se, agg_first):
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

        results = fit.GelbachDecompositionResults.tidy()
        results = results.query("panels == 'Levels (units)'")
        coefficients = results.coefficients
        ci_lower = results.ci_lower
        ci_upper = results.ci_upper

        filtered_df = stata_results.query(f"model == '{model}' and se == '{se}'")

        for g in ["g1", "g2", "explained_effect"]:
            coef_diff = filtered_df.xs(g).Coefficient - coefficients.xs(g)
            lower_diff = filtered_df.xs(g)["CI Lower"] - ci_lower.xs(g)
            upper_diff = filtered_df.xs(g)["CI Upper"] - ci_upper.xs(g)

            assert np.all(np.abs(coef_diff) < 1e-6), (
                f"Failed for {g} with values from Stata of {filtered_df.xs(g).Coefficient} and Python of {coefficients.xs(g)}"
            )
            if False:
                assert np.all(np.abs(lower_diff) < 1e-4), (
                    f"Failed for {g} with values {filtered_df.xs(g)['CI Lower']} and {ci_lower.xs(g)}"
                )
                assert np.all(np.abs(upper_diff) < 1e-4), (
                    f"Failed for {g} with values {filtered_df.xs(g)['CI Upper']} and {ci_upper.xs(g)}"
                )

    decompose_and_compare(
        fit=fit,
        stata_results=stata_results,
        param="x1",
        combine_covariates=combine_covariates,
        seed=3,
        reps=100,
        model="model 1"
        if combine_covariates == {"g1": ["x21", "x22"], "g2": ["x23"]}
        else "model 2",
        se=se,
        cluster="cluster" if se == "cluster" else None,
        agg_first=agg_first,
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

    for key, value in fit1.GelbachDecompositionResults.results.absolute.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.results.absolute.get(key)
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

    for key, value in fit1.GelbachDecompositionResults.results.absolute.items():
        assert (
            value - fit2.GelbachDecompositionResults.results.absolute.get(key) < 1e-08
        ), (
            f"Failed for {key} with values {value} and {fit2.GelbachDecompositionResults.results.absolute.get(key)}"
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

    for key, value in fit1.GelbachDecompositionResults.results.absolute.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.results.absolute.get(key)
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

    for key, value in fit1.GelbachDecompositionResults.results.absolute.items():
        np.testing.assert_allclose(
            value, fit2.GelbachDecompositionResults.results.absolute.get(key)
        )


@pytest.mark.parametrize(
    "combine_config,x1_vars",
    [
        ({"g1": ["x21", "x22"], "g2": ["x23"]}, None),
        ({"mixed": ["x21"], "other": ["x22", "x23"]}, None),
        ({"all": ["x21", "x22", "x23"]}, None),
        ({"g1": ["x22"], "g2": ["x23"]}, ["x21"]),
        ({"remaining": ["x22", "x23"]}, ["x21"]),
        ({"g1": ["x23"]}, ["x21", "x22"]),
        ({"single": ["x23"]}, ["x21", "x22"]),
    ],
)
def test_agg_first_equivalence(combine_config, x1_vars):
    """
    Test that agg_first=True and agg_first=False produce identical tidy() DataFrames.
    Tests both scenarios with and without x1_vars (background controls).
    """
    data = gelbach_data(nobs=150)

    results = {}
    for agg_first in [True, False]:
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

        decomp_kwargs = {
            "decomp_var": "x1",
            "combine_covariates": combine_config,
            "seed": 123,
            "agg_first": agg_first,
            "only_coef": True,
        }

        if x1_vars is not None:
            decomp_kwargs["x1_vars"] = x1_vars

        fit.decompose(**decomp_kwargs)
        results[agg_first] = fit.GelbachDecompositionResults.tidy()

    pd.testing.assert_frame_equal(
        results[True],
        results[False],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
        obj=f"tidy() DataFrames for agg_first=True vs agg_first=False with combine_config={combine_config}, x1_vars={x1_vars}",
    )


def smoke_test_only_coef():
    data = pf.get_data()
    fit = pf.feols("Y~X1 + X2 | f1", data=data)
    fit.decompose(param="X1", only_coef=True)


@pytest.mark.parametrize("agg_first", [True, False])
def test_x1_vars(agg_first):
    "Test Gelbach decomposition with x1_vars."
    data = pd.read_csv("tests/data/gelbach.csv")

    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    fit.decompose(
        param="x1",
        x1_vars=["x21"],
        seed=3,
        only_coef=True,
        agg_first=agg_first,
        combine_covariates={"ALL": ["x22", "x23"]},
    )
    # test that param ALL is .5024257
    np.testing.assert_allclose(
        fit.GelbachDecompositionResults.results.absolute["ALL"], 0.5024257
    )

    fit.decompose(
        param="x1",
        x1_vars=["x21", "x22"],
        combine_covariates={"ALL": ["x23"]},
        seed=3,
        agg_first=agg_first,
        only_coef=True,
    )
    np.testing.assert_allclose(
        fit.GelbachDecompositionResults.results.absolute["ALL"], 0.3149754
    )

    fit.decompose(
        param="x1",
        x1_vars="x21+x22",
        combine_covariates={"ALL": ["x23"]},
        seed=3,
        agg_first=agg_first,
        only_coef=True,
    )
    np.testing.assert_allclose(
        fit.GelbachDecompositionResults.results.absolute["ALL"], 0.3149754
    )


def test_tidy_snapshot(gelbach_decomposition):
    "Mock test for tidy()."
    tidy_result = gelbach_decomposition.tidy(alpha=0.05).query(
        "panels == 'Levels (units)'"
    )
    tidy_result.round(6).to_string()

    return tidy_result


@pytest.mark.parametrize(
    "panels",
    ["all", "levels", "share_full", "share_explained", ["levels", "share_full"]],
)
@pytest.mark.parametrize("caption", [None, "Test Caption"])
@pytest.mark.parametrize("column_heads", [None, ["Total", "Direct", "Mediated"]])
@pytest.mark.parametrize("use_panel_heads", [False, True])
@pytest.mark.parametrize("rgroup_sep", [None, "tb", "t", "b", ""])
@pytest.mark.parametrize("add_notes", [None, "Custom test note"])
def test_etable_snapshot(
    gelbach_decomposition,
    panels,
    caption,
    column_heads,
    use_panel_heads,
    rgroup_sep,
    add_notes,
):
    if use_panel_heads:
        if panels == "all":
            panel_heads = ["Absolute", "Share of Total", "Share of Explained"]
        elif panels == ["levels", "share_full"]:
            panel_heads = ["Absolute", "Share of Total"]
        else:
            panel_heads = ["Custom Panel"]
    else:
        panel_heads = None

    gelbach_decomposition.etable(
        panels=panels,
        caption=caption,
        column_heads=column_heads,
        panel_heads=panel_heads,
        rgroup_sep=rgroup_sep,
        add_notes=add_notes,
        digits=3,
        type="gt",
    ).as_raw_html()

    assert 0 == 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
@pytest.mark.parametrize(
    "data_transform",
    [
        "base",  # No transformation
        "neg_y_plus_2x23",  # y = -y + 2*x23
        "tiny_f1_effect",  # y += 0.0001 * f1 (tests small effects)
        "neg_y",  # y = -y
    ],
)
@pytest.mark.parametrize("annotate_shares", [True, False])
def test_coefplot_comprehensive(data_transform, annotate_shares):
    "Comprehensive smoke test for coefplot covering all parameters and data transformations."
    rng = np.random.default_rng(12345)
    data = gelbach_data(nobs=200)
    data["f1"] = rng.normal(size=200)

    if data_transform == "neg_y_plus_2x23":
        data["y"] = -data["y"] + 2 * data["x23"]
    elif data_transform == "tiny_f1_effect":
        data["y"] += 0.0001 * data["f1"]
    elif data_transform == "neg_y":
        data["y"] = -data["y"]

    fit = pf.feols("y ~ x1 + x21 + x22 + x23 + f1", data=data, demeaner_backend="rust")
    gb = fit.decompose(  # type: ignore[attr-defined]
        decomp_var="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22", "x23", "f1"]},
        only_coef=True,
    )

    gb.coefplot(annotate_shares=annotate_shares)
    plt.close("all")

    gb.coefplot(
        title="Custom Test Title", figsize=(10, 6), annotate_shares=annotate_shares
    )
    plt.close("all")

    gb.coefplot(components_order=["g2", "g1"], annotate_shares=annotate_shares)
    plt.close("all")

    gb.coefplot(keep=["g1", "g2"], annotate_shares=annotate_shares)
    plt.close("all")

    gb.coefplot(drop=["f1"], annotate_shares=annotate_shares)
    plt.close("all")

    gb.coefplot(keep=["g1"], exact_match=True, annotate_shares=annotate_shares)
    plt.close("all")

    gb.coefplot(
        labels={"g1": "Group One", "g2": "Group Two"}, annotate_shares=annotate_shares
    )
    plt.close("all")

    gb.coefplot(
        notes="Custom test note for validation", annotate_shares=annotate_shares
    )
    plt.close("all")

    gb.coefplot(
        components_order=["g2", "g1"],
        title="Complex Test",
        figsize=(14, 8),
        keep=["g1", "g2"],
        labels={"g1": "First Group", "g2": "Second Group"},
        notes="Testing multiple parameters together",
        annotate_shares=annotate_shares,
    )
    plt.close("all")
