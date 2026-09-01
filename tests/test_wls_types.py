import numpy as np
import pandas as pd
import pytest

import pyfixest as pf


def _make_frequency_weight_data():
    """Return aggregate rows and their literal frequency-weight expansion."""
    x = np.array([-1.5, -0.5, 0.5, 1.5, -1.4, -0.4, 0.6, 1.4, -1.6, -0.6, 0.4, 1.6])
    z = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    first_stage_error = np.array(
        [0.3, -0.2, 0.1, -0.1, -0.25, 0.2, -0.15, 0.05, 0.12, -0.18, 0.22, -0.08]
    )
    structural_error = np.array(
        [0.2, -0.4, 0.3, -0.1, -0.25, 0.35, -0.2, 0.15, 0.28, -0.32, 0.18, -0.12]
    )
    fixef = np.repeat(["a", "b", "c"], 4)
    fixef_effect = np.repeat([-0.8, 0.4, 1.1], 4)
    count = np.array([1, 3, 2, 4, 2, 1, 3, 2, 4, 1, 2, 3])
    endogenous = 1.1 * z + 0.25 * x + 0.15 * fixef_effect + first_stage_error
    response = 1.0 + 1.4 * endogenous - 0.6 * x + fixef_effect + structural_error

    aggregate = pd.DataFrame(
        {
            "y": response,
            "x": x,
            "d": endogenous,
            "z": z,
            "fe": fixef,
            "count": count,
        }
    )
    expanded = (
        aggregate.loc[aggregate.index.repeat(count)]
        .drop(columns="count")
        .reset_index(drop=True)
    )
    return aggregate, expanded


def _assert_fit_equal(fit1, fit2, vcov_types, rtol=1e-5):
    "Assert that two regression fits have identical coefficients, vcov, and SEs."
    assert fit1._N == fit2._N, "Number of observations is not the same."

    np.testing.assert_allclose(
        fit1.coef().values, fit2.coef().values, rtol=rtol, err_msg="Coefficients differ"
    )

    for vcov_type in vcov_types:
        fit1_vcov = fit1.vcov(vcov_type)
        fit2_vcov = fit2.vcov(vcov_type)

        np.testing.assert_allclose(
            fit1_vcov._vcov,
            fit2_vcov._vcov,
            rtol=rtol,
            err_msg=f"Vcov differs for {vcov_type}",
        )
        np.testing.assert_allclose(
            fit1_vcov.se().values,
            fit2_vcov.se().values,
            rtol=rtol,
            err_msg=f"SEs differ for {vcov_type}",
        )


@pytest.mark.parametrize(
    "fml,cols,vcov_types",
    [
        # Without fixed effects - test hetero vcov types
        ("Y ~ X1", ["Y", "X1"], ["iid", "HC1", "HC2", "HC3"]),
        # Without fixed effects - test CRV (need to include cluster var in aggregation)
        ("Y ~ X1", ["Y", "X1", "f1"], [{"CRV1": "f1"}, {"CRV3": "f1"}]),
        # With fixed effects - HC2/HC3 not supported
        (
            "Y ~ X1 | f1",
            ["Y", "X1", "f1"],
            ["iid", "HC1", {"CRV1": "f1"}, {"CRV3": "f1"}],
        ),
    ],
)
def test_fweights_ols(fml, cols, vcov_types):
    """Test that fweights are correctly implemented for OLS models."""
    data = pf.get_data(model="Fepois")

    # Drop rows with NaN in columns used for aggregation to ensure same N
    data = data.dropna(subset=cols)

    data_agg = (
        data[cols].groupby(cols).size().reset_index().rename(columns={0: "count"})
    )

    fit_raw = pf.feols(fml, data=data, vcov="iid")
    fit_agg = pf.feols(
        fml,
        data=data_agg,
        weights="count",
        weights_type="fweights",
        vcov="iid",
    )

    _assert_fit_equal(fit_raw, fit_agg, vcov_types)


@pytest.mark.parametrize(
    "fml,vcov_types,performance_attributes",
    [
        ("y ~ x", ["iid", "HC1", "HC2", "HC3"], ["_rmse", "_r2", "_adj_r2"]),
        (
            "y ~ x | fe",
            ["iid", "HC1"],
            ["_rmse", "_r2", "_adj_r2", "_r2_within", "_adj_r2_within"],
        ),
    ],
)
def test_fweights_ols_matches_literal_expansion(
    fml, vcov_types, performance_attributes
):
    """Frequency-weighted OLS state matches literally repeated observations."""
    aggregate, expanded = _make_frequency_weight_data()
    counts = aggregate["count"].to_numpy(dtype=np.int64)

    fit_weighted = pf.feols(
        fml,
        data=aggregate,
        weights="count",
        weights_type="fweights",
        vcov="iid",
    )
    fit_expanded = pf.feols(fml, data=expanded, vcov="iid")

    _assert_fit_equal(fit_weighted, fit_expanded, vcov_types, rtol=1e-10)
    assert len(expanded) == fit_weighted._N
    assert fit_weighted._N_rows == len(aggregate)

    np.testing.assert_allclose(
        np.repeat(fit_weighted.resid(), counts),
        fit_expanded.resid(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted.resid(),
        fit_weighted._within_data.response.flatten()
        - fit_weighted._X @ fit_weighted._beta_hat,
        rtol=1e-10,
        atol=1e-10,
    )

    for attribute in performance_attributes:
        np.testing.assert_allclose(
            getattr(fit_weighted, attribute),
            getattr(fit_expanded, attribute),
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize("estimator", ["ols", "iv", "poisson"])
def test_fweights_singleton_detection_matches_literal_expansion(estimator):
    """An aggregate row with count above one is not an FE singleton."""
    aggregate = pd.DataFrame(
        {
            "y": [2, 1, 4, 2, 5],
            "x": [0.2, -1.0, 1.0, -0.5, 0.8],
            "d": [0.1, -0.8, 1.2, -0.2, 1.0],
            "z": [0.3, -1.1, 0.7, -0.7, 1.3],
            "fe": ["solo", "a", "a", "b", "b"],
            "count": [2, 1, 2, 1, 2],
        }
    )
    expanded = (
        aggregate.loc[aggregate.index.repeat(aggregate["count"])]
        .drop(columns="count")
        .reset_index(drop=True)
    )
    formula = {
        "ols": "y ~ x | fe",
        "iv": "y ~ x + [d ~ z] | fe",
        "poisson": "y ~ x | fe",
    }[estimator]
    fit = pf.fepois if estimator == "poisson" else pf.feols
    fit_kwargs = (
        {"iwls_tol": 1e-10, "iwls_maxiter": 100} if estimator == "poisson" else {}
    )

    weighted = fit(
        formula,
        data=aggregate,
        weights="count",
        weights_type="fweights",
        vcov="iid",
        **fit_kwargs,
    )
    literal = fit(formula, data=expanded, vcov="iid", **fit_kwargs)

    assert weighted._N_rows == len(aggregate)
    assert len(expanded) == weighted._N
    assert literal._N_rows == len(expanded)
    np.testing.assert_allclose(weighted.coef(), literal.coef(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        weighted._vcov,
        literal._vcov,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "fml,fe_col",
    [
        ("Y ~ X1", None),
        ("Y ~ X1 | f1", "f1"),
    ],
)
def test_fweights_poisson(fml, fe_col):
    """Test that fweights are correctly implemented for Poisson models."""
    data = pf.get_data(model="Fepois")

    cols = ["Y", "X1"]
    if fe_col:
        cols.append(fe_col)

    data_agg = (
        data[cols].groupby(cols).size().reset_index().rename(columns={0: "count"})
    )

    fit_raw = pf.fepois(fml, data=data, vcov="iid", iwls_tol=1e-12, iwls_maxiter=100)
    fit_agg = pf.fepois(
        fml,
        data=data_agg,
        weights="count",
        weights_type="fweights",
        vcov="iid",
        iwls_tol=1e-12,
        iwls_maxiter=100,
    )

    # Poisson only supports HC1 for hetero-robust SEs
    vcov_types = ["iid", "hetero"]
    _assert_fit_equal(fit_raw, fit_agg, vcov_types)


@pytest.mark.parametrize("fml", ["y ~ x + [d ~ z]", "y ~ x + [d ~ z] | fe"])
def test_fweights_iv_matches_literal_expansion(fml):
    """Frequency-weighted IV and first-stage state match repeated observations."""
    aggregate, expanded = _make_frequency_weight_data()
    counts = aggregate["count"].to_numpy(dtype=np.int64)

    fit_weighted = pf.feols(
        fml,
        data=aggregate,
        weights="count",
        weights_type="fweights",
        vcov="hetero",
    )
    fit_expanded = pf.feols(fml, data=expanded, vcov="hetero")

    np.testing.assert_allclose(
        fit_weighted.coef().values,
        fit_expanded.coef().values,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted._vcov, fit_expanded._vcov, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        np.repeat(fit_weighted.resid(), counts),
        fit_expanded.resid(),
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        fit_weighted._pi_hat, fit_expanded._pi_hat, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        np.repeat(fit_weighted._X_hat, counts),
        fit_expanded._X_hat,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.repeat(fit_weighted._v_hat, counts),
        fit_expanded._v_hat,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted._X_hat,
        fit_weighted._model_1st_stage._X @ fit_weighted._pi_hat,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted._v_hat,
        fit_weighted._model_1st_stage._within_data.response.flatten()
        - fit_weighted._X_hat,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted._f_stat_1st_stage,
        fit_expanded._f_stat_1st_stage,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        fit_weighted._p_value_1st_stage,
        fit_expanded._p_value_1st_stage,
        rtol=1e-10,
        atol=1e-10,
    )

    fit_weighted.IV_Diag()
    fit_expanded.IV_Diag()
    np.testing.assert_allclose(
        fit_weighted._eff_F, fit_expanded._eff_F, rtol=1e-10, atol=1e-10
    )


def test_aweights():
    data = pf.get_data()
    data["weights"] = np.ones(data.shape[0])

    fit1 = pf.feols("Y ~ X1", data=data)
    fit2 = pf.feols("Y ~ X1", data=data, weights_type="aweights")
    fit3 = pf.feols("Y ~ X1", data=data, weights="weights", weights_type="aweights")

    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)
    np.testing.assert_allclose(fit1.tidy().values, fit3.tidy().values)
