import numpy as np
import pandas as pd
import polars as pl
import pytest

from pyfixest.estimation import feols, fepois
from pyfixest.report.utils import (
    rename_categoricals,
    rename_event_study_coefs,
)
from pyfixest.utils.utils import capture_context, get_data, ssc


def test_multicol_overdetermined_iv():
    data = get_data()
    fit = feols(
        fml="Y ~ X2 +  f1| f1 | X1 ~ Z1 + Z2",
        data=data,
        ssc=ssc(k_adj=False),
        vcov={"CRV1": "f1"},
    )

    assert fit.collinearity.dropped_coef_names == ("f1",)
    assert fit.collinearity_instruments.dropped_coef_names == ("f1",)

    np.testing.assert_allclose(
        fit._beta_hat, np.array([-0.174227, -0.993607], dtype=float), rtol=1e-5
    )
    np.testing.assert_allclose(
        fit.coeftable.se, np.array([0.018416, 0.104009]), rtol=1e-5
    )


@pytest.mark.parametrize(
    ("cluster", "explicit_cluster", "fml", "weights", "weights_type"),
    [
        ("f1:f2", "joint2", "Y ~ X1 | f1", "observation_weight", "aweights"),
        ("f1:f2", "joint2", "Y ~ X1 | f1", "frequency_weight", "fweights"),
        ("f1:f2:f3", "joint3", "Y ~ X1 + [X2 ~ Z1] | f1", None, "aweights"),
        ("  f1  :\t f2 + f3  ", "joint2 + f3", "Y ~ X1", None, "aweights"),
    ],
)
def test_cluster_interactions_match_explicit_groups(
    cluster, explicit_cluster, fml, weights, weights_type
):
    data = get_data(N=400, seed=1576).dropna().copy()
    data["f3"] = np.arange(len(data)) % 4
    data["observation_weight"] = np.linspace(0.5, 2.0, len(data))
    data["frequency_weight"] = np.arange(len(data)) % 4 + 1
    for name, columns in (
        ("joint2", ["f1", "f2"]),
        ("joint3", ["f1", "f2", "f3"]),
    ):
        data[name] = pd.factorize(pd.MultiIndex.from_frame(data[columns]))[0]

    fit = feols(
        fml,
        data=data,
        weights=weights,
        weights_type=weights_type,
        vcov={"CRV1": cluster},
    )
    reference = feols(
        fml,
        data=data,
        weights=weights,
        weights_type=weights_type,
        vcov={"CRV1": explicit_cluster},
    )

    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        reference.variance_covariance.vcov,
        rtol=1e-10,
        atol=1e-10,
        err_msg="cluster-interaction covariance differs from explicit joint groups",
    )
    assert fit.variance_covariance.G == reference.variance_covariance.G
    assert fit.variance_covariance.df_t == reference.variance_covariance.df_t
    assert fit.variance_covariance.cluster_ids is not None
    assert fit.variance_covariance.cluster_ids.shape == (
        fit.sample_info.n_rows,
        len(fit.variance_covariance.G),
    )


def test_cluster_interaction_ids_follow_post_estimation_vcov_data():
    data = get_data(N=180, seed=1576).dropna().copy()
    fit = feols("Y ~ X1", data=data, vcov="iid")
    alternate = fit._data.copy()
    alternate["f2"] = np.roll(alternate["f2"].to_numpy(), 1)

    fit.vcov({"CRV1": "f1:f2"}, data=alternate)
    expected = pd.factorize(pd.MultiIndex.from_frame(alternate[["f1", "f2"]]))[0]
    np.testing.assert_array_equal(fit.variance_covariance.cluster_ids[:, 0], expected)
    np.testing.assert_array_equal(fit._cluster_array("f1:f2"), expected)
    assert "f1:f2" not in data.columns

    fit.vcov("iid")
    assert fit.variance_covariance.cluster_ids is None


def test_cluster_interactions_with_poisson_and_multiple_estimation():
    data = get_data(N=300, seed=1576).dropna().copy()
    data["joint"] = pd.factorize(pd.MultiIndex.from_frame(data[["f1", "f2"]]))[0]

    multi = feols("sw(Y, Y2) ~ X1 | f1", data=data, vcov={"CRV1": "f1:f2"})
    multi_reference = feols("sw(Y, Y2) ~ X1 | f1", data=data, vcov={"CRV1": "joint"})
    for fit, reference in zip(
        multi.all_fitted_models.values(),
        multi_reference.all_fitted_models.values(),
        strict=True,
    ):
        np.testing.assert_allclose(
            fit.variance_covariance.vcov,
            reference.variance_covariance.vcov,
            rtol=1e-10,
            atol=1e-10,
            err_msg="multiple-estimation covariance differs for joint groups",
        )

    poisson_data = get_data(N=300, seed=1576, model="Fepois").dropna().copy()
    poisson_data["joint"] = pd.factorize(
        pd.MultiIndex.from_frame(poisson_data[["f1", "f2"]])
    )[0]
    poisson = fepois("Y ~ X1 | f1", data=poisson_data, vcov={"CRV1": "f1:f2"})
    poisson_reference = fepois("Y ~ X1 | f1", data=poisson_data, vcov={"CRV1": "joint"})
    np.testing.assert_allclose(
        poisson.variance_covariance.vcov,
        poisson_reference.variance_covariance.vcov,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Poisson covariance differs for joint groups",
    )


def test_crv3_cluster_interaction_matches_explicit_groups():
    data = get_data(N=120, seed=1576).dropna().copy()
    data["f1"] = np.arange(len(data)) % 2
    data["f2"] = np.arange(len(data)) // 2 % 3
    data["joint"] = pd.factorize(pd.MultiIndex.from_frame(data[["f1", "f2"]]))[0]

    fit = feols("Y ~ X1", data=data, vcov={"CRV3": "f1:f2"})
    reference = feols("Y ~ X1", data=data, vcov={"CRV3": "joint"})
    np.testing.assert_allclose(
        fit.variance_covariance.vcov,
        reference.variance_covariance.vcov,
        rtol=1e-10,
        atol=1e-10,
        err_msg="CRV3 covariance differs for joint groups",
    )


def test_cluster_interaction_post_estimation_uses_stored_ids():
    data = get_data(N=160, seed=1576).dropna().copy()
    n_rows = len(data)
    data["f1"] = np.arange(n_rows) % 2
    data["f2"] = np.arange(n_rows) // 2 % 2
    data["D"] = np.arange(n_rows) // 4 % 2
    data["joint"] = pd.factorize(pd.MultiIndex.from_frame(data[["f1", "f2"]]))[0]

    fit = feols("Y ~ D", data=data, vcov={"CRV1": "f1  :  f2"})
    reference = feols("Y ~ D", data=data, vcov={"CRV1": "joint"})

    bootstrap = fit.wildboottest(reps=15, param="D", seed=17)
    bootstrap_reference = reference.wildboottest(reps=15, param="D", seed=17)
    for quantity in ("t value", "Pr(>|t|)", "ssc"):
        np.testing.assert_allclose(
            bootstrap[quantity],
            bootstrap_reference[quantity],
            err_msg=f"wild bootstrap {quantity} differs for joint groups",
        )

    ccv = fit.ccv(treatment="D", pk=0.5, qk=0.5, n_splits=4, seed=17)
    ccv_reference = reference.ccv(treatment="D", pk=0.5, qk=0.5, n_splits=4, seed=17)
    pd.testing.assert_frame_equal(ccv, ccv_reference)


def test_polars_input():
    data = get_data()
    data_pl = pl.from_pandas(data)
    fit = feols("Y ~ X1", data=data)
    fit.predict(newdata=data_pl)

    data = get_data(model="Fepois")
    data["offset"] = np.log(np.random.default_rng(0).uniform(0.5, 3.0, len(data)))
    data_pl = pl.from_pandas(data)
    fit = fepois("Y ~ X1", data=data_pl)
    fit_offset = fepois("Y ~ X1", data=data, offset="offset")
    fit_offset.predict(newdata=data_pl)


def test_integer_XY():
    # Create a random number generator
    rng = np.random.default_rng()

    N = 1000
    X = rng.normal(0, 1, N)
    f = rng.choice([0, 1], N)
    Y = 2 * X + rng.normal(0, 1, N) + f * 2
    Y = np.round(Y).astype(np.int64)
    X = np.round(X).astype(np.int64)

    df = pd.DataFrame({"Y": Y, "X": X, "f": f})

    fit1 = feols("Y ~ X | f", data=df, vcov="iid")
    fit2 = feols("Y ~ X + C(f)", data=df)

    np.testing.assert_allclose(fit1.coef().xs("X"), fit2.coef().xs("X"))


def test_coef_update():
    rng = np.random.default_rng(1234)
    data = get_data().dropna(subset=["Y", "X1", "X2"])
    data_subsample = data.sample(frac=0.5, random_state=1234)
    m = feols("Y ~ X1 + X2", data=data_subsample)
    new_points_id = rng.choice(
        data.index.difference(data_subsample.index), 5, replace=False
    )
    X_new, y_new = (
        np.c_[
            np.ones(len(new_points_id)), data.loc[new_points_id][["X1", "X2"]].values
        ],
        data.loc[new_points_id]["Y"].values,
    )
    updated_coefs = m.update(X_new, y_new)
    full_coefs = (
        feols(
            "Y ~ X1 + X2",
            data=data.loc[data_subsample.index.append(pd.Index(new_points_id))],
        )
        .coef()
        .values
    )

    np.testing.assert_allclose(updated_coefs, full_coefs)


def test_rename_categoricals():
    coefnames = ["C(var)[T.1]", "C(var)[T.2]", "C(var2)[T.1]", "C(var2)[T.2]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "C(var)[T.1]": "var::1",
        "C(var)[T.2]": "var::2",
        "C(var2)[T.1]": "var2::1",
        "C(var2)[T.2]": "var2::2",
    }

    # with strings:
    coefnames = ["Intercept", "C(f4)[T.B]", "C(f4)[T.C]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "C(f4)[T.B]": "f4::B",
        "C(f4)[T.C]": "f4::C",
    }

    # with reference levels:
    coefnames = [
        "Intercept",
        "C(f4, contr.treatment(base='A'))[T.B]",
        "C(f4, contr.treatment(base='A'))[T.C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "C(f4, contr.treatment(base='A'))[T.B]": "f4::B",
        "C(f4, contr.treatment(base='A'))[T.C]": "f4::C",
    }

    # without 'T.' in the categorical notation:
    coefnames = [
        "C(f4)[B]",
        "C(f4)[C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "C(f4)[B]": "f4::B",
        "C(f4)[C]": "f4::C",
    }

    # without C() and no 'T.' notation
    coefnames = [
        "f4[B]",
        "f4[C]",
    ]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "f4[B]": "f4::B",
        "f4[C]": "f4::C",
    }

    # with categoricals:
    coefnames = ["Intercept", "variable1[T.value1]", "variable1[T.value2]"]
    renamed = rename_categoricals(coefnames)
    assert renamed == {
        "Intercept": "Intercept",
        "variable1[T.value1]": "variable1::value1",
        "variable1[T.value2]": "variable1::value2",
    }

    # Test with labels
    coefnames = ["C(variable1)[T.value1]", "variable2[T.value2]"]
    labels = {"variable1": "var1", "variable2": "var2"}
    renamed = rename_categoricals(coefnames, labels=labels)
    assert renamed == {
        "C(variable1)[T.value1]": "var1::value1",
        "variable2[T.value2]": "var2::value2",
    }

    # Test with custom template
    coefnames = ["C(variable1)[T.value1]", "variable2[T.value2]"]
    template = "{variable}--{value}"
    renamed = rename_categoricals(coefnames, template=template)
    assert renamed == {
        "C(variable1)[T.value1]": "variable1--value1",
        "variable2[T.value2]": "variable2--value2",
    }


def test_rename_event_study_coefs():
    coefnames = [
        "C(rel_year, contr.treatment(base=-1.0))[T.-20.0]",
        "C(rel_year, contr.treatment(base=-1.0))[T.-19.0]",
        "Intercept",
    ]

    renamed = rename_event_study_coefs(coefnames)
    assert renamed == {
        "C(rel_year, contr.treatment(base=-1.0))[T.-20.0]": "rel_year::-20.0",
        "C(rel_year, contr.treatment(base=-1.0))[T.-19.0]": "rel_year::-19.0",
        "Intercept": "Intercept",
    }


def _foo():
    "Simulate a callable for testing context capture behavior."
    ...


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ i(f1) | f2",
        "Y ~ i(f1, ref=1.0) | f2",
        "Y ~ i(f1, X1) | f2",
        "Y ~ i(f1, X1) + X2 | f2",
    ],
)
def test_predict_newdata_i_transform(fml):
    """Test predict(newdata=...) works for models using the i() transform."""
    data = get_data(N=500, seed=42).dropna()
    newdata = data.iloc[:100]

    fit = feols(fml, data=data)
    pred_full = fit.predict()
    pred_new = fit.predict(newdata=newdata)

    assert pred_full.shape[0] == fit.sample_info.n_obs
    assert pred_new.shape[0] == len(newdata)


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ poly(X1, 2)",
        "Y ~ poly(X1, 2) | f1",
        "Y ~ poly(X1, 2) + X2 | f1 + f2",
    ],
)
def test_predict_newdata_poly_transform(fml):
    """Test predict(newdata=...) works for models using poly()."""
    data = get_data(N=500, seed=42).dropna()
    newdata = data.iloc[:100]

    fit = feols(fml, data=data)
    pred_full = fit.predict()
    pred_new = fit.predict(newdata=newdata)

    assert pred_full.shape[0] == fit.sample_info.n_obs
    assert pred_new.shape[0] == len(newdata)


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1 | f1:f2",
        "Y ~ X1 + X2 | f1:f2",
    ],
)
def test_predict_newdata_fe_interaction(fml):
    """Test predict(newdata=...) works for fixed-effect interactions."""
    data = get_data(N=500, seed=42).dropna()
    newdata = data.iloc[:100]

    fit = feols(fml, data=data)
    pred_full = fit.predict()
    pred_new = fit.predict(newdata=newdata)

    assert pred_full.shape[0] == fit.sample_info.n_obs
    assert pred_new.shape[0] == len(newdata)


@pytest.mark.parametrize(
    "fml",
    [
        "Y ~ X1 + C(f1)",
        "Y ~ X1 + i(f1)",
        "Y ~ X1 + i(f1, X2)",
        "Y ~ X1 + C(f1) | f2",
    ],
)
def test_predict_newdata_unseen_category(fml):
    """
    Rows whose categorical level was not seen during fitting must predict NaN.

    Otherwise formulaic/i() silently encode the unseen level as the reference
    level, yielding a finite-but-wrong prediction.
    """
    data = get_data(N=500, seed=42).dropna()
    newdata = data.iloc[:100].copy()
    newdata.iloc[0, newdata.columns.get_loc("f1")] = 999999.0  # unseen level

    fit = feols(fml, data=data)
    pred_new = fit.predict(newdata=newdata)

    assert np.isnan(pred_new[0]), "unseen categorical level should predict NaN"
    assert np.all(np.isfinite(pred_new[1:])), "seen rows should remain finite"


def test_i_bin_bin2_separate_state():
    """i(a, b, bin=..., bin2=...) must store separate bin mappings per variable.

    Regression test: previously _apply_binning used a single "bin_mapping" key
    in shared encoder state, so the second variable reused the first's mapping.
    """
    data = get_data(N=500, seed=42).dropna()
    data["f1"] = data["f1"].astype(int).astype(str)
    data["f2"] = data["f2"].astype(int).astype(str)

    fit = feols(
        "Y ~ i(f1, f2, bin={'low': ['0', '1']}, bin2={'hi': ['0', '1']})",
        data=data,
    )
    coefnames = [str(c) for c in fit._coefnames]
    f1_binned = any("low" in c for c in coefnames)
    f2_binned = any("hi" in c for c in coefnames)
    assert f1_binned, f"f1 should be binned to 'low', got: {coefnames}"
    assert f2_binned, f"f2 should be binned to 'hi', got: {coefnames}"


def test_predict_decoy_column_not_flagged_unseen():
    """A data column named like a contrast keyword arg must not cause NaN.

    Regression test: _categorical_levels used regex to extract identifiers from
    factor expressions, so with a formula like `C(f1, contr.treatment(base=1.0))`
    the keyword-arg name `base` was treated as a model variable. A continuous
    data column named `base` was then checked against f1's categories and
    flagged every row as unseen -> all predictions NaN. The formula must
    contain the keyword argument for this test to pin the bug.
    """
    data = get_data(N=500, seed=42).dropna()
    data["base"] = np.random.default_rng(99).normal(size=len(data))
    fit = feols("Y ~ X1 + C(f1, contr.treatment(base=1.0))", data=data)
    pred = fit.predict(newdata=data.iloc[:100])
    assert np.all(np.isfinite(pred)), "decoy column should not cause NaN predictions"


def test_predict_binned_i_not_flagged_unseen():
    """predict(newdata=...) with binned i() must not flag valid raw levels as unseen.

    Regression test: _rows_with_unseen_categories checked raw newdata values
    against post-binning categories, so a valid raw level like 'a' (binned to
    'low') was flagged unseen -> NaN prediction.
    """
    data = pd.DataFrame(
        {
            "Y": np.random.default_rng(0).normal(size=200),
            "f1": np.random.default_rng(2).choice(["a", "b", "c", "d"], size=200),
        }
    )
    fit = feols("Y ~ i(f1, bin={'low': ['a', 'b']})", data=data)
    pred = fit.predict(newdata=data.iloc[:50])
    assert np.all(np.isfinite(pred)), "valid binned levels should not be NaN"

    # Truly unseen level should still be NaN
    newdata = data.iloc[:10].copy()
    newdata.iloc[0, newdata.columns.get_loc("f1")] = "zzz"
    pred_unseen = fit.predict(newdata=newdata)
    assert np.isnan(pred_unseen[0]), "truly unseen level should predict NaN"
    assert np.all(np.isfinite(pred_unseen[1:])), "seen rows should remain finite"


def test_context_capture():
    # `_foo` is in caller's stack frame, if should be captured
    # call with -1 to account for adding one more frame inside the function
    context = capture_context(-1)
    assert "_foo" in context

    # `_foo` is in caller's stack frame, but we ask for a deeper stack, `_foo` should not be captured
    context = capture_context(1)
    assert "_foo" not in context

    context = capture_context({})
    assert context == {}

    context = capture_context({"_foo": _foo})
    assert context == {"_foo": _foo}


def _fixef_test_data(n=500, with_nan=False, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Y": rng.normal(size=n),
            "X1": rng.normal(size=n),
            "g": rng.choice(["north", "south", "east"], n),
            "h": rng.choice(["u", "v", "w", "x"], n),
        }
    )
    if with_nan:
        df.loc[df.index[:10], "g"] = np.nan
    return df


def test_fixef_returns_tidy_coefficients():
    """fixef() must return tidy coefficients with decoded labels.

    Regression test: the predict/fixef rewrite briefly returned internal
    encodings (keys like `__fixed_effect__(g)` with ngroup codes as levels).
    """
    df = _fixef_test_data()
    fit = feols("Y ~ X1 | g + h", data=df)
    coefficients = fit.fixef(atol=1e-12, btol=1e-12)

    assert list(coefficients.columns) == [
        "variable",
        "code",
        "level",
        "coefficient",
    ]
    assert set(coefficients["variable"]) == {"g", "h"}
    assert set(coefficients.loc[coefficients["variable"].eq("g"), "level"]) == {
        "north",
        "south",
        "east",
    }
    assert set(coefficients.loc[coefficients["variable"].eq("h"), "level"]) == {
        "u",
        "v",
        "w",
        "x",
    }
    assert not coefficients["code"].isna().any()

    # reference normalization: the second FE carries a zero reference level,
    # the first FE (spanning the intercept) does not
    assert (
        coefficients.loc[coefficients["variable"].eq("h"), "coefficient"].eq(0.0).any()
    )

    # The returned table cannot be used to mutate prediction state.
    coefficients.loc[:, "coefficient"] = np.nan

    # predict(newdata=fit data) must reproduce in-sample predictions,
    # exercising the internal tidy table used for FE mapping
    np.testing.assert_allclose(
        fit.predict(), fit.predict(newdata=df), rtol=1e-6, atol=1e-8
    )


def test_fixef_nan_fe_level_excluded():
    """NaN FE values in the fit data must not surface as a 'nan' level."""
    df = _fixef_test_data(with_nan=True)
    fit = feols("Y ~ X1 | g", data=df)
    coefficients = fit.fixef()

    assert set(coefficients["level"]) == {"north", "south", "east"}
    assert not coefficients["level"].str.lower().eq("nan").any()


def test_fixef_interacted_labels():
    """Interacted FEs decode to `g:h` keys with `val1,val2` level labels."""
    df = _fixef_test_data()
    fit = feols("Y ~ X1 | g:h", data=df)
    coefficients = fit.fixef(atol=1e-12, btol=1e-12)

    assert fit._fml == "Y ~ X1 | g:h"
    assert fit._fixef == "g:h"
    assert coefficients["variable"].unique().tolist() == ["g:h"]
    levels = set(coefficients["level"])
    assert all("," in level for level in levels)
    observed = {f"{g},{h}" for g, h in zip(df["g"], df["h"], strict=True)}
    assert levels == observed

    np.testing.assert_allclose(
        fit.predict(), fit.predict(newdata=df), rtol=1e-6, atol=1e-8
    )

    newdata = df.iloc[:2].copy()
    newdata.iloc[0, newdata.columns.get_loc("g")] = "unseen"
    with pytest.warns(UserWarning, match=r"fixed effect `g:h`"):
        prediction = fit.predict(newdata=newdata)
    assert np.isnan(prediction[0])


def test_fixef_excludes_singleton_levels_from_prediction():
    """Singleton FE levels are unavailable, not zero-valued references."""
    df = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 2.0, 4.0, 3.0],
            "X1": [0.0, 1.0, 0.0, 1.0, 2.0],
            "g": ["a", "a", "b", "b", "c"],
            "h": ["u", "u", "v", "v", "w"],
        }
    )

    with pytest.warns(UserWarning, match="1 singleton fixed effect"):
        fit = feols("Y ~ X1 | g:h", data=df)

    prediction = fit.predict(newdata=df)
    fixed_effects = fit.fixef()

    assert np.isfinite(prediction[:-1]).all()
    assert np.isnan(prediction[-1])
    assert set(fixed_effects["level"]) == {"a,u", "b,v"}
