import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

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


def _weighted_gelbach_data() -> pd.DataFrame:
    rng = np.random.default_rng(20240626)
    nobs = 80
    x1 = rng.normal(size=nobs)
    x21 = rng.normal(size=nobs)
    x22 = rng.binomial(1, 0.4, size=nobs)
    x23 = rng.normal(size=nobs)
    weights = rng.uniform(0.6, 2.4, size=nobs)
    y = (
        1.0
        + 0.75 * x1
        + 0.35 * x21
        - 0.25 * x22
        + 0.2 * x23
        + 0.15 * x1 * x21
        + rng.normal(scale=0.25, size=nobs)
    )
    return pd.DataFrame(
        {
            "Y": y,
            "x1": x1,
            "x21": x21,
            "x22": x22,
            "x23": x23,
            "f1": np.arange(nobs) % 8,
            "w": weights,
            "fw": 1 + np.arange(nobs) % 3,
        }
    )


def _manual_aweighted_gelbach_levels(
    data: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    names = ["Intercept", "x1", "x21", "x22", "x23"]
    x_raw = np.column_stack(
        [
            np.ones(data.shape[0]),
            data["x1"].to_numpy(),
            data["x21"].to_numpy(),
            data["x22"].to_numpy(),
            data["x23"].to_numpy(),
        ]
    )
    y_raw = data["Y"].to_numpy()
    weights_sqrt = np.sqrt(data["w"].to_numpy())
    x = x_raw * weights_sqrt[:, None]
    y = y_raw * weights_sqrt

    mask = np.array([name != "x1" for name in names])
    x1 = x[:, ~mask]
    x1 = np.column_stack([weights_sqrt, x1])
    x2 = x[:, mask]
    mediator_names = [name for name, keep in zip(names, mask, strict=True) if keep]
    group_indices = {
        name: [mediator_names.index(covariate) for covariate in covariates]
        for name, covariates in groups.items()
    }

    beta_short = np.linalg.lstsq(x1, y, rcond=None)[0]
    beta_full = np.linalg.lstsq(x, y, rcond=None)[0]
    beta2 = beta_full[mask]
    x1_inv = np.linalg.pinv(x1.T @ x1)
    x_inv = np.linalg.pinv(x.T @ x)
    gamma_matrix = x1_inv @ x1.T @ x2
    gamma = gamma_matrix[1, :]

    mediator_effects = {
        name: float(np.sum(gamma[variable_idx] * beta2[variable_idx]))
        for name, variable_idx in group_indices.items()
    }
    direct_effect = float(beta_short[1])
    full_effect = float(beta_full[1])
    explained_effect = sum(mediator_effects.values())
    estimates = {
        "direct_effect": direct_effect,
        "full_effect": full_effect,
        "explained_effect": explained_effect,
        "unexplained_effect": direct_effect - explained_effect,
        **mediator_effects,
    }

    short_resid = y - x1 @ beta_short
    full_resid = y - x @ beta_full
    short_weight = x1 @ x1_inv[:, 1]
    full_weight = x @ x_inv[:, 1]
    beta2_indices = np.flatnonzero(mask)

    mediator_group_if = {}
    for name, variable_idx in group_indices.items():
        group_gamma = gamma[variable_idx]
        group_beta2 = beta2[variable_idx]
        group_beta2_weight = x_inv[:, beta2_indices[variable_idx]] @ group_gamma
        beta2_if = (x @ group_beta2_weight) * full_resid
        group_auxiliary_fit = gamma_matrix[:, variable_idx] @ group_beta2
        group_h = x2[:, variable_idx] @ group_beta2
        group_auxiliary_resid = group_h - x1 @ group_auxiliary_fit
        gamma_if = short_weight * group_auxiliary_resid
        mediator_group_if[name] = beta2_if + gamma_if

    explained_if = np.sum(np.column_stack(list(mediator_group_if.values())), axis=1)
    influence_df = pd.DataFrame(
        {
            "direct_effect": short_weight * short_resid,
            "full_effect": full_weight * full_resid,
            "explained_effect": explained_if,
            "unexplained_effect": short_weight * short_resid - explained_if,
            **mediator_group_if,
        }
    )

    rank = np.linalg.matrix_rank(x.T @ x)
    df = max(data.shape[0] - rank, 1)
    hc1_factor = data.shape[0] / df
    std_error = np.sqrt(hc1_factor * np.square(influence_df).sum(axis=0))
    crit = np.abs(t.ppf(0.05 / 2, df))
    estimates_series = pd.Series(estimates, dtype=float)

    return pd.DataFrame(
        {
            "coefficients": estimates_series,
            "std_error": std_error.reindex(estimates_series.index),
            "ci_lower": estimates_series
            - crit * std_error.reindex(estimates_series.index),
            "ci_upper": estimates_series
            + crit * std_error.reindex(estimates_series.index),
        }
    )


@pytest.fixture
def b1x2_results():
    """Results generated by Stata's b1x2 package in tests/data/gelbach.txt."""
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
def test_against_stata_b1x2(b1x2_results, combine_covariates, se, agg_first):
    data = pd.read_stata("tests/data/gelbach.dta")
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    def decompose_and_compare(
        fit,
        b1x2_results,
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
            inference="bootstrap" if cluster is not None else "analytic",
        )

        results = fit.GelbachDecompositionResults.tidy()
        results = results.query("panels == 'Levels (units)'")
        coefficients = results.coefficients
        ci_lower = results.ci_lower
        ci_upper = results.ci_upper

        filtered_df = b1x2_results.query(f"model == '{model}' and se == '{se}'")

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
        b1x2_results=b1x2_results,
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
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        digits=6,
        inference="bootstrap",
        reps=3,
        seed=123,
    )
    # cluster set in decompose
    fit2.decompose(
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        digits=6,
        cluster="cluster",
        inference="bootstrap",
        reps=3,
        seed=123,
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
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        digits=6,
        only_coef=True,
    )
    fit2.decompose(
        param="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        digits=6,
        only_coef=True,
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


def test_decompose_defaults_to_analytic_inference():
    data = gelbach_data(nobs=150)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(decomp_var="x1")
    tidy = gb.tidy().query("panels == 'Levels (units)'")

    assert gb.inference == "analytic"
    assert {"std_error", "ci_lower", "ci_upper"}.issubset(tidy.columns)
    assert np.all(np.isfinite(tidy["std_error"]))


def test_decompose_bootstrap_inference_preserved():
    data = gelbach_data(nobs=100)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(decomp_var="x1", inference="bootstrap", reps=5, seed=123)
    tidy = gb.tidy().query("panels == 'Levels (units)'")

    assert gb.inference == "bootstrap"
    assert {"ci_lower", "ci_upper"}.issubset(tidy.columns)
    assert "std_error" not in tidy.columns


def test_only_coef_has_no_inference_columns():
    data = gelbach_data(nobs=100)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(decomp_var="x1", only_coef=True)
    tidy = gb.tidy()

    assert "std_error" not in tidy.columns
    assert "ci_lower" not in tidy.columns
    assert "ci_upper" not in tidy.columns


def test_analytic_matches_only_coef_point_estimates():
    data = gelbach_data(nobs=150)
    fit_analytic = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
    fit_only_coef = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    analytic = fit_analytic.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
    ).tidy()
    only_coef = fit_only_coef.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        only_coef=True,
    ).tidy()

    np.testing.assert_allclose(analytic["coefficients"], only_coef["coefficients"])


def test_analytic_ci_changes_with_alpha():
    data = gelbach_data(nobs=150)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(decomp_var="x1")
    ci_95 = gb.tidy(alpha=0.05).query("panels == 'Levels (units)'")
    ci_90 = gb.tidy(alpha=0.10).query("panels == 'Levels (units)'")

    assert np.any(np.abs(ci_95["ci_lower"] - ci_90["ci_lower"]) > 1e-12)
    assert np.any(np.abs(ci_95["ci_upper"] - ci_90["ci_upper"]) > 1e-12)


@pytest.mark.parametrize("agg_first", [True, False])
def test_analytic_inference_with_x1_vars_and_combine_covariates(agg_first):
    data = pd.read_csv("tests/data/gelbach.csv")
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(
        decomp_var="x1",
        x1_vars=["x21"],
        combine_covariates={"ALL": ["x22", "x23"]},
        agg_first=agg_first,
    )
    tidy = gb.tidy().query("panels == 'Levels (units)'")

    assert {"std_error", "ci_lower", "ci_upper"}.issubset(tidy.columns)
    assert np.isfinite(tidy.loc["ALL", "std_error"])


def test_analytic_agg_first_equivalent_standard_errors():
    data = gelbach_data(nobs=150)
    results = {}

    for agg_first in [True, False]:
        fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)
        results[agg_first] = fit.decompose(
            decomp_var="x1",
            combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
            agg_first=agg_first,
        ).tidy()

    pd.testing.assert_frame_equal(
        results[True],
        results[False],
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_etable_analytic_df():
    data = gelbach_data(nobs=100)
    fit = pf.feols("y ~ x1 + x21 + x22 + x23", data=data)

    gb = fit.decompose(decomp_var="x1")
    table_df = gb.etable(type="df")

    assert isinstance(table_df, pd.DataFrame)
    assert "" in table_df.index.get_level_values(-1)


@pytest.mark.parametrize(
    "panels",
    ["all", "levels", "share_full", "share_explained", ["levels", "share_full"]],
)
@pytest.mark.parametrize("caption", [None, "Test Caption"])
@pytest.mark.parametrize("column_heads", [None, ["Total", "Direct", "Mediated"]])
@pytest.mark.parametrize("use_panel_heads", [False, True])
# @pytest.mark.parametrize("rgroup_sep", [None, "tb", "t", "b", ""])
# @pytest.mark.parametrize("add_notes", [None, "Custom test note"])
def test_etable_snapshot(
    gelbach_decomposition,
    panels,
    caption,
    column_heads,
    use_panel_heads,
    # rgroup_sep,
    # add_notes,
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
        # rgroup_sep=rgroup_sep,
        # add_notes=add_notes,
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

    fit = pf.feols(
        "y ~ x1 + x21 + x22 + x23 + f1",
        data=data,
        demeaner=pf.MapDemeaner(backend="rust"),
    )
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
        title="Complex Test",
        figsize=(14, 8),
        keep=["g1", "g2"],
        labels={"g1": "First Group", "g2": "Second Group"},
        notes="Testing multiple parameters together",
        annotate_shares=annotate_shares,
    )
    plt.close("all")


@pytest.mark.parametrize(
    "fml", ["Y ~ x1 + x21 + x22 + x23", "Y ~ x1 + x21 + x22 | x23"]
)
def test_weights(fml):
    rng = np.random.default_rng(123)
    N = 1000
    Y = rng.choice(range(10), N)
    x1 = rng.choice(range(2), N)
    x21 = rng.choice(range(2), N)
    x22 = rng.choice(range(2), N)
    x23 = rng.choice(range(5), N)

    data = pd.DataFrame({"Y": Y, "x1": x1, "x21": x21, "x22": x22, "x23": x23})
    agg_vars = ["Y", "x1", "x21", "x22", "x23"]

    data_agg = data.groupby(agg_vars).size().reset_index().rename(columns={0: "count"})

    fit = pf.feols(fml=fml, data=data)
    fit_agg = pf.feols(
        fml=fml,
        data=data_agg,
        weights="count",
        weights_type="fweights",
        # demeaner_backend="rust",
    )

    # test that coefs() are identical
    np.testing.assert_allclose(fit.coef(), fit_agg.coef())

    # decomposition without combine_covariates:
    decompse_kwargs_1 = {"param": "x1", "only_coef": True}
    # with combine covariates 1:
    decompse_kwargs_2 = {
        "param": "x1",
        "only_coef": True,
        "combine_covariates": {"g1": ["x21"], "g2": ["x22"], "g3": re.compile("x23")},
    }
    # with combine covariates 2:
    decompse_kwargs_3 = {
        "param": "x1",
        "only_coef": True,
        "combine_covariates": {"g1": ["x21", "x22"], "g2": re.compile("x23")},
    }

    for kwargs in [decompse_kwargs_1, decompse_kwargs_2, decompse_kwargs_3]:
        gb = fit.decompose(**kwargs)
        gb_agg = fit_agg.decompose(**kwargs)
        tidy_orig = gb.tidy()
        tidy_agg = gb_agg.tidy()
        np.testing.assert_allclose(
            tidy_orig.select_dtypes(include=[np.number]),
            tidy_agg.select_dtypes(include=[np.number]),
        )


@pytest.mark.parametrize("weights_type", ["aweights", "fweights"])
def test_weighted_point_estimates_against_stata_b1x2(weights_type):
    reference_path = Path("tests/data/gelbach_b1x2_weighted.csv")
    if not reference_path.exists():
        pytest.skip("Run tests/data/gelbach_b1x2_weighted.do in Stata first.")

    data = pd.read_stata("tests/data/gelbach.dta")
    observation = np.arange(1, data.shape[0] + 1)
    data["aw"] = 0.75 + np.mod(observation, 7) / 4
    data["fw"] = 1 + np.mod(observation, 3)
    weight_column = "aw" if weights_type == "aweights" else "fw"
    groups = {"g1": ["x21", "x22"], "g2": ["x23"]}

    reference = pd.read_csv(reference_path)
    reference = reference.query("weights_type == @weights_type").set_index("effect")

    results = {}
    for agg_first in [True, False]:
        results[agg_first] = (
            pf.feols(
                "y ~ x1 + x21 + x22 + x23",
                data=data,
                weights=weight_column,
                weights_type=weights_type,
            )
            .decompose(
                decomp_var="x1",
                combine_covariates=groups,
                agg_first=agg_first,
                only_coef=True,
            )
            .tidy(panels="levels")["coefficients"]
        )

        np.testing.assert_allclose(
            results[agg_first].reindex(reference.index),
            reference["coefficient"],
            rtol=1e-8,
            atol=1e-9,
        )

    pd.testing.assert_series_equal(
        results[True],
        results[False],
        check_exact=False,
        rtol=1e-8,
        atol=1e-9,
    )


def _assert_valid_analytic_levels(result: pd.DataFrame) -> None:
    inference_columns = ["std_error", "ci_lower", "ci_upper"]
    assert set(inference_columns).issubset(result.columns)
    assert np.isfinite(result[inference_columns].to_numpy()).all()
    assert (result["std_error"] >= 0).all()
    assert (result["ci_lower"] <= result["coefficients"]).all()
    assert (result["coefficients"] <= result["ci_upper"]).all()


def test_analytic_weighted_point_estimates_match_manual_wls_reference():
    data = _weighted_gelbach_data()
    assert not float(data["w"].sum()).is_integer()
    groups = {"x21": ["x21"], "x22": ["x22"], "x23": ["x23"]}

    fit = pf.feols(
        fml="Y ~ x1 + x21 + x22 + x23",
        data=data,
        weights="w",
        weights_type="aweights",
    )
    result = fit.decompose(
        decomp_var="x1",
        only_coef=True,
    ).tidy(panels="levels")

    reference = _manual_aweighted_gelbach_levels(data, groups)
    np.testing.assert_allclose(
        result["coefficients"],
        reference["coefficients"],
        rtol=1e-9,
        atol=1e-10,
    )


def test_analytic_weighted_inference_matches_manual_wls_reference():
    data = _weighted_gelbach_data()
    groups = {"g1": ["x21"], "g2": ["x22", "x23"]}

    fit = pf.feols(
        fml="Y ~ x1 + x21 + x22 + x23",
        data=data,
        weights="w",
        weights_type="aweights",
    )
    result = fit.decompose(
        decomp_var="x1",
        combine_covariates=groups,
    ).tidy(panels="levels")

    reference = _manual_aweighted_gelbach_levels(data, groups)
    np.testing.assert_allclose(
        result[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        reference[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        rtol=1e-9,
        atol=1e-10,
    )


def test_analytic_weighted_inference_with_fixed_effects_returns_finite_results():
    data = _weighted_gelbach_data()

    fit = pf.feols(
        fml="Y ~ x1 + x21 + x22 | f1",
        data=data,
        weights="w",
        weights_type="aweights",
    )
    result = fit.decompose(
        decomp_var="x1",
        combine_covariates={"g1": ["x21"], "g2": ["x22"]},
    ).tidy(panels="levels")

    numeric = result[["coefficients", "std_error", "ci_lower", "ci_upper"]]
    assert np.isfinite(numeric.to_numpy()).all()


@pytest.mark.parametrize(
    "fml", ["Y ~ x1 + x21 + x22 + x23", "Y ~ x1 + x21 + x22 | x23"]
)
@pytest.mark.parametrize("agg_first", [True, False])
def test_frequency_weighted_analytic_inference_matches_expanded_data(fml, agg_first):
    rng = np.random.default_rng(123)
    nobs = 1000
    data = pd.DataFrame(
        {
            "Y": rng.choice(range(10), nobs),
            "x1": rng.choice(range(2), nobs),
            "x21": rng.choice(range(2), nobs),
            "x22": rng.choice(range(2), nobs),
            "x23": rng.choice(range(5), nobs),
        }
    )
    aggregate_columns = ["Y", "x1", "x21", "x22", "x23"]
    compressed_data = data.groupby(aggregate_columns).size().reset_index(name="count")

    def decompose(fit):
        return fit.decompose(
            decomp_var="x1",
            combine_covariates={
                "g1": ["x21", "x22"],
                "g2": re.compile("x23"),
            },
            agg_first=agg_first,
        ).tidy()

    expanded = (
        decompose(pf.feols(fml=fml, data=data)).rename_axis("effect").reset_index()
    )
    compressed = (
        decompose(
            pf.feols(
                fml=fml,
                data=compressed_data,
                weights="count",
                weights_type="fweights",
            )
        )
        .rename_axis("effect")
        .reset_index()
    )

    sort_columns = ["panels", "effect"]
    expanded = expanded.sort_values(sort_columns).reset_index(drop=True)
    compressed = compressed.sort_values(sort_columns).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        expanded[sort_columns],
        compressed[sort_columns],
    )
    np.testing.assert_allclose(
        expanded[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        compressed[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        rtol=1e-8,
        atol=1e-9,
    )


def test_constant_aweights_match_unweighted_point_estimates():
    data = _weighted_gelbach_data()
    data["constant_aw"] = 2.5
    groups = {"g1": ["x21", "x22"], "g2": ["x23"]}

    unweighted = (
        pf.feols("Y ~ x1 + x21 + x22 + x23", data=data)
        .decompose(
            decomp_var="x1",
            combine_covariates=groups,
            only_coef=True,
        )
        .tidy(panels="levels")
    )
    weighted = (
        pf.feols(
            "Y ~ x1 + x21 + x22 + x23",
            data=data,
            weights="constant_aw",
            weights_type="aweights",
        )
        .decompose(
            decomp_var="x1",
            combine_covariates=groups,
        )
        .tidy(panels="levels")
    )

    np.testing.assert_allclose(
        weighted["coefficients"],
        unweighted["coefficients"],
        rtol=1e-9,
        atol=1e-10,
    )
    _assert_valid_analytic_levels(weighted)


def test_constant_fweights_match_expanded_analytic_results():
    data = _weighted_gelbach_data()
    data["constant_fw"] = 3
    expanded_data = data.loc[data.index.repeat(data["constant_fw"])].reset_index(
        drop=True
    )
    groups = {"g1": ["x21", "x22"], "g2": ["x23"]}

    expanded = (
        pf.feols("Y ~ x1 + x21 + x22 + x23", data=expanded_data)
        .decompose(
            decomp_var="x1",
            combine_covariates=groups,
        )
        .tidy(panels="levels")
    )
    compressed = (
        pf.feols(
            "Y ~ x1 + x21 + x22 + x23",
            data=data,
            weights="constant_fw",
            weights_type="fweights",
        )
        .decompose(
            decomp_var="x1",
            combine_covariates=groups,
        )
        .tidy(panels="levels")
    )

    np.testing.assert_allclose(
        compressed[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        expanded[["coefficients", "std_error", "ci_lower", "ci_upper"]],
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize("weights_type", [None, "aweights", "fweights"])
def test_grouped_mediator_effects_equal_individual_sums(weights_type):
    data = _weighted_gelbach_data()
    groups = {"group_a": ["x21", "x22"], "group_b": ["x23"]}
    fit_kwargs = {}
    if weights_type is not None:
        fit_kwargs = {
            "weights": "w" if weights_type == "aweights" else "fw",
            "weights_type": weights_type,
        }

    individual = (
        pf.feols("Y ~ x1 + x21 + x22 + x23", data=data, **fit_kwargs)
        .decompose(
            decomp_var="x1",
            only_coef=True,
        )
        .tidy(panels="levels")
    )
    grouped = (
        pf.feols("Y ~ x1 + x21 + x22 + x23", data=data, **fit_kwargs)
        .decompose(
            decomp_var="x1",
            combine_covariates=groups,
            agg_first=True,
            only_coef=True,
        )
        .tidy(panels="levels")
    )

    np.testing.assert_allclose(
        grouped.loc["group_a", "coefficients"],
        individual.loc[["x21", "x22"], "coefficients"].sum(),
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        grouped.loc["group_b", "coefficients"],
        individual.loc["x23", "coefficients"],
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        grouped.loc[
            [
                "direct_effect",
                "full_effect",
                "explained_effect",
                "unexplained_effect",
            ],
            "coefficients",
        ],
        individual.loc[
            [
                "direct_effect",
                "full_effect",
                "explained_effect",
                "unexplained_effect",
            ],
            "coefficients",
        ],
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.parametrize("weights_type", ["aweights", "fweights"])
def test_weighted_analytic_inference_is_well_formed(weights_type):
    data = _weighted_gelbach_data()
    weight_column = "w" if weights_type == "aweights" else "fw"

    result = (
        pf.feols(
            "Y ~ x1 + x21 + x22 + x23",
            data=data,
            weights=weight_column,
            weights_type=weights_type,
        )
        .decompose(
            decomp_var="x1",
            combine_covariates={"g1": ["x21", "x22"], "g2": ["x23"]},
        )
        .tidy(panels="levels")
    )

    _assert_valid_analytic_levels(result)
