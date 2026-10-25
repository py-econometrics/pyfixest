"""Check the deprecation shims that keep 0.60 fitted-model attributes readable.

Every row of `DEPRECATED_ATTRIBUTES` is exercised on a fit of each class it
names. The values the shims rebuild are compared with pyfixest 0.60.0 in
`test_release_contract.py`; the checks here pin the 0.60 identities between
the square-root-weighted arrays, which only a weighted fit distinguishes from
the unweighted typed state.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import astuple

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import PyfixestDeprecationWarning
from pyfixest.estimation.internals.retention import omitted_attributes
from pyfixest.estimation.models._deprecated_attributes import (
    DEPRECATED_ATTRIBUTES,
    Alias,
    Removed,
)
from pyfixest.estimation.models.feglm_ import Feglm
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.quantreg.quantreg_ import Quantreg

MODEL_CLASSES = {cls.__name__: cls for cls in (Feols, Feiv, Feglm, Fepois, Quantreg)}

# Rows whose 0.60 value is a rebuilt form of the replacement (a square-root
# weighting, a list or `None` default, a reshape, or padding) rather than the
# replacement itself. The release contract compares them with 0.60.0.
REBUILT = {
    "get_performance",
    "_sample_split_var",
    "_N_rows",
    "_n_fe",
    "_fixef",
    "_icovars",
    "_ssc_dict",
    "_weights",
    "_Y",
    "_X",
    "_Z",
    "_u_hat",
    "_clustervar",
    "_G",
    "_ssc",
    "deviance",
    "_loglik_null",
    "_pseudo_r2",
    "_sumFE",
    "_X_hat",
    "_v_hat",
    "_non_exo_instruments",
    "_p_iv",
    "_collin_vars",
    "_collin_index",
    "_collin_vars_z",
    "_collin_index_z",
    "_support_iid_inference",
}

EVENT_STUDY_ROWS = {"_res_cohort_eventtime_dict", "_yname", "_gname", "_tname"}
EVENT_STUDY_ROWS |= {"_idname", "_att"}

ENTRIES = [
    pytest.param(attribute, class_name, entry, id=f"{class_name}.{attribute.name}")
    for attribute in DEPRECATED_ATTRIBUTES
    for class_name, entry in attribute.classes.items()
]
ALIASES = [param for param in ENTRIES if isinstance(param.values[2], Alias)]
REMOVED = [param for param in ENTRIES if isinstance(param.values[2], Removed)]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    data = pf.get_data(N=500, seed=2026).dropna().reset_index(drop=True)
    rng = np.random.default_rng(2026)
    data["w"] = rng.uniform(0.5, 3.0, data.shape[0])
    data["Y_bin"] = (data["Y"] > 0).astype(int)
    data["Y_count"] = rng.poisson(np.exp(0.3 * data["X1"]))
    return data


@pytest.fixture(scope="module")
def fits(data: pd.DataFrame) -> dict[str, Feols]:
    """One fit per model class, with every lazily published state set."""
    ols = pf.feols("Y ~ X1 + X2 | f1", data, weights="w", vcov={"CRV1": "f1"})
    ols.fixef()
    ri = pf.feols("Y ~ X1 + X2", data)
    ri.ritest(
        "X1",
        reps=9,
        rng=np.random.default_rng(2026),
        store_ritest_statistics=True,
        choose_algorithm="slow",
    )
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data, weights="w", vcov={"CRV1": "f1"})
    iv.IV_Diag()
    glm = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit", vcov="hetero")
    glm.fixef(atol=1e-12, btol=1e-12)
    poisson = pf.fepois("Y_count ~ X1 + X2 | f1", data, weights="w", vcov="hetero")
    poisson.fixef()
    quantile = pf.quantreg("Y ~ X1 + X2", data, quantile=0.4, vcov="nid")
    return {
        "Feols": ols,
        "Feiv": iv,
        "Feglm": glm,
        "Fepois": poisson,
        "Quantreg": quantile,
        "ritest": ri,
    }


@pytest.fixture(scope="module")
def event_study() -> Feols:
    data = pd.read_csv("pyfixest/did/data/df_het.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pf.event_study(
            data,
            yname="dep_var",
            idname="unit",
            tname="year",
            gname="g",
            estimator="saturated",
        )


@pytest.fixture(scope="module")
def split_fit(data: pd.DataFrame) -> Feols:
    return pf.feols("Y ~ X1", data, split="f2").to_list()[0]


def _fit_for(name, class_name, fits, event_study, split_fit):
    if name.startswith("_ritest_"):
        return fits["ritest"]
    if name in EVENT_STUDY_ROWS:
        return event_study
    if name == "_sample_split_value":
        return split_fit
    return fits[class_name]


def _assert_same(value, expected, *, name: str) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(value, expected, err_msg=name)
    else:
        assert value == expected or value is expected, name


def test_entries_install_on_model_classes():
    """Every entry installs a descriptor on the model class it names."""
    for attribute in DEPRECATED_ATTRIBUTES:
        for class_name in attribute.classes:
            assert class_name in MODEL_CLASSES, attribute.name
            assert attribute.name in vars(MODEL_CLASSES[class_name]), attribute.name
    names = [attribute.name for attribute in DEPRECATED_ATTRIBUTES]
    assert len(names) == len(set(names))


def test_maketables_metadata_stays_silent(fits, split_fit, data):
    """Released maketables reads these names, including on split models."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyfixestDeprecationWarning)
        full_sample = pf.feols("Y ~ X1", data, fsplit="f2").to_list()[0]
        for fit in (fits["Feols"], split_fit, full_sample):
            assert fit._adj_r2 == fit.fitstat.adj_r2
            split = fit.model.sample_split
            assert fit._sample_split_var == (None if split is None else split.var)
            if split is not None:
                assert fit._sample_split_value == (
                    "all" if split.value is None else split.value
                )


@pytest.mark.parametrize(("attribute", "class_name", "entry"), ALIASES)
def test_alias_returns_replacement(
    attribute, class_name, entry, fits, event_study, split_fit
):
    fit = _fit_for(attribute.name, class_name, fits, event_study, split_fit)
    if attribute.silent:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = getattr(fit, attribute.name)
    else:
        with pytest.warns(PyfixestDeprecationWarning) as record:
            value = getattr(fit, attribute.name)
        (warning,) = record
        assert str(warning.message) == (
            f"`fit.{attribute.name}` is deprecated and will be removed in a future "
            f"release; use `{entry.replacement}` instead."
        )
        # stacklevel points at the attribute access in user code
        assert warning.filename == __file__

    replacement = eval(entry.replacement, {"fit": fit})  # names a real field
    if attribute.name not in REBUILT:
        _assert_same(value, replacement, name=attribute.name)


@pytest.mark.parametrize(("attribute", "class_name", "entry"), REMOVED)
def test_removed_attribute_names_replacement(
    attribute, class_name, entry, fits, event_study, split_fit
):
    fit = _fit_for(attribute.name, class_name, fits, event_study, split_fit)
    with pytest.raises(AttributeError, match="Sorry") as excinfo:
        getattr(fit, attribute.name)
    assert f"`{entry.replacement}`" in str(excinfo.value)
    assert not hasattr(fit, attribute.name)


@pytest.mark.parametrize("class_name", MODEL_CLASSES)
def test_k_fe_is_removed(class_name, fits):
    """All estimators reject the encoded Series, including inherited access."""
    fit = fits[class_name]
    with pytest.raises(
        AttributeError, match=r"_k_fe.*removed.*fit\.fixef_counts\.n_levels_by_fe"
    ):
        fit._k_fe  # noqa: B018


def test_unpublished_state_stays_missing(data):
    """Deprecated names raise `AttributeError` until their state is published."""
    fit = pf.feols("Y ~ X1 | f1", data)
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1]", data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in (
            "_alpha",
            "_sumFE",
            "_sample_split_value",
            "_yname",
            "_ritest_statistics",
            "_ritest_sample_stat",
            "_ritest_pvalue",
        ):
            assert not hasattr(fit, name), name
        with pytest.raises(AttributeError, match=r"IV_Diag\(\)"):
            iv._eff_F  # noqa: B018
    with pytest.raises(AttributeError, match=re.escape("fit.coeftable.se")):
        fit._se = np.zeros(1)


@pytest.mark.parametrize("class_name", MODEL_CLASSES)
def test_get_performance_deprecated_call(class_name, fits):
    fit = fits[class_name]
    expected = fit.fitstat
    with pytest.warns(
        PyfixestDeprecationWarning, match=r"get_performance.*fit\.fitstat"
    ):
        assert fit.get_performance() is None
    np.testing.assert_array_equal(
        astuple(fit.fitstat),
        astuple(expected),
        err_msg="get_performance fit statistics",
    )


def test_weighted_ols_rebuilds_square_root_weighted_arrays(fits, data):
    fit = fits["Feols"]
    sqrt_w = np.sqrt(data["w"].to_numpy())
    within = fit.within_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        X, Y, Z = fit._X, fit._Y, fit._Z
        np.testing.assert_allclose(X, within.design * sqrt_w[:, None])
        np.testing.assert_allclose(Y, within.response * sqrt_w[:, None])
        np.testing.assert_allclose(Z, X)
        np.testing.assert_allclose(fit._weights.flatten(), data["w"])
        # 0.60 identities between the stored arrays
        u_hat_weighted = Y.flatten() - X @ fit._beta_hat
        np.testing.assert_allclose(u_hat_weighted, sqrt_w * fit.resid())
        np.testing.assert_allclose(fit._u_hat, u_hat_weighted)
        np.testing.assert_allclose(fit._scores, X * u_hat_weighted[:, None])
        np.testing.assert_allclose(fit._hessian, X.T @ X)
        np.testing.assert_allclose(fit._tZX, X.T @ X)
        np.testing.assert_allclose(fit._bread @ fit._hessian, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(fit._sumFE, sqrt_w * fit.fixef_estimates.sumFE)


def test_weighted_iv_rebuilds_square_root_weighted_arrays(fits, data):
    fit = fits["Feiv"]
    sqrt_w = np.sqrt(data["w"].to_numpy())
    within = fit.within_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        X, Y, Z = fit._X, fit._Y, fit._Z
        np.testing.assert_allclose(X, within.design * sqrt_w[:, None])
        np.testing.assert_allclose(Y, within.response * sqrt_w[:, None])
        np.testing.assert_allclose(Z, within.instruments * sqrt_w[:, None])
        # 0.60: the 2SLS residual on the weighted scale
        np.testing.assert_allclose(
            Y.flatten() - X @ fit._beta_hat, sqrt_w * fit.resid()
        )
        np.testing.assert_allclose(fit._u_hat, sqrt_w * fit.resid())
        # 0.60: the weighted first stage splits the endogenous regressor
        np.testing.assert_allclose(
            fit._X_hat + fit._v_hat, sqrt_w * within.endogenous.flatten()
        )
        assert fit._non_exo_instruments == ["Z1"]
        assert fit._p_iv == 1


def test_glm_rebuilds_working_weighted_arrays(fits, data):
    """0.60 GLMs stored the final IRLS arrays weighted by sqrt(working weights)."""
    weighted_glm = pf.feglm(
        "Y_bin ~ X1 + X2 | f1", data, family="logit", weights="w", vcov="hetero"
    )
    weighted_glm.fixef(atol=1e-12, btol=1e-12)
    for fit in (fits["Feglm"], fits["Fepois"], weighted_glm):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyfixestDeprecationWarning)
            X, Y, u_hat = fit._X, fit._Y.flatten(), fit._u_hat
            np.testing.assert_allclose(u_hat, Y - X @ fit._beta_hat, atol=1e-10)
            np.testing.assert_allclose(fit._scores, X * u_hat[:, None], atol=1e-10)
            np.testing.assert_allclose(fit._hessian, X.T @ X)
            np.testing.assert_allclose(
                fit._weights.flatten(), fit.working_state.working_weights
            )
            assert fit._weights.shape == (fit.sample_info.n_rows, 1)
            np.testing.assert_array_equal(
                fit._Z, X, err_msg="GLM _Z is the solver design"
            )
            if not isinstance(fit, Fepois):
                # Weighted non-Poisson GLMs postdate 0.60: this is the intended
                # current contract, not a claim of historical weighted parity.
                # Fixed effects live on link scale; only weighted fits scale
                # the legacy sum by sqrt(final IRLS weights).
                sum_fe = fit.fitted_values.link - (
                    fit.model_matrix.independent.to_numpy() @ fit.coef().to_numpy()
                )
                if fit.options.has_weights:
                    sum_fe *= np.sqrt(fit.working_state.working_weights)
                # FE recovery uses an iterative solver, so allow its residual error.
                np.testing.assert_allclose(
                    fit._sumFE,
                    sum_fe,
                    atol=1e-8,
                    err_msg="GLM _sumFE link-scale identity",
                )
                assert fit._sumFE.shape == (fit.sample_info.n_rows,)
                np.testing.assert_array_equal(
                    fit._Y_hat_link, fit.fitted_values.link, err_msg="GLM _Y_hat_link"
                )
                np.testing.assert_allclose(
                    fit._Y_hat_response,
                    1 / (1 + np.exp(-fit._Y_hat_link)),
                    err_msg="logit _Y_hat_response is inverse link",
                )
                assert not np.allclose(fit._Y_hat_link, fit._Y_hat_response)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        assert fits["Fepois"]._Y.shape == (fits["Fepois"].sample_info.n_rows, 1)
        assert fits["Feglm"]._Y.ndim == 1
        # weighted Poisson fits had no null model in 0.60
        assert fits["Fepois"]._loglik_null is None
        assert fits["Fepois"]._pseudo_r2 is None


def test_cluster_counts_keep_min_padding(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        oneway = pf.feols("Y ~ X1", data, vcov={"CRV1": "f1"})
        G = oneway.variance_covariance.G[0]
        assert oneway._G == [G, G, G]
        assert oneway._ssc.shape == (1, 1)
        conventional = pf.feols(
            "Y ~ X1 | f2 + f1",
            data,
            vcov={"CRV1": "f1+f2"},
            ssc=pf.ssc(k_adj=False, k_fixef="full", G_adj=False, G_df="conventional"),
        )
        assert list(conventional.variance_covariance.G) == conventional._G
        assert len(conventional._G) == 3
        multiway = pf.feols(
            "Y ~ X1", data, vcov={"CRV1": "f1+f2"}, ssc=pf.ssc(G_df="min")
        )
        counts = [
            data.f1.nunique(),
            data.f2.nunique(),
            len(data[["f1", "f2"]].drop_duplicates()),
        ]
        # Padding must choose a minimum, not the first count.
        assert counts[0] > counts[1]
        assert [min(counts)] * 3 == multiway._G
        unclustered = pf.feols("Y ~ X1", data, vcov="hetero")
        assert unclustered._G == []
        for fit, fixef, clusters, expected_ssc in (
            (
                conventional,
                "f2+f1",
                ["f1", "f2"],
                {
                    "k_adj": False,
                    "k_fixef": "full",
                    "G_adj": False,
                    "G_df": "conventional",
                },
            ),
            (
                unclustered,
                None,
                None,
                {"k_adj": True, "k_fixef": "nonnested", "G_adj": True, "G_df": "min"},
            ),
        ):
            assert fit._fixef == fixef
            assert fit._clustervar == clusters
            assert fit._ssc_dict == expected_ssc
            assert isinstance(fit._ssc_dict, dict)
        assert isinstance(conventional._clustervar, list)


def test_legacy_defaults_and_capabilities(fits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        for name in ("Feols", "Feiv", "Quantreg"):
            assert fits[name].deviance is None, name
        for name in ("Feglm", "Fepois"):
            assert np.isfinite(fits[name].deviance), name
            assert fits[name].deviance == fits[name].fitstat.deviance, name
        for name in ("Feiv", "Feglm", "Fepois"):
            assert fits[name]._support_iid_inference is True, name
        assert fits["Feols"]._support_crv3_inference is True
        assert fits["Feglm"]._support_crv3_inference is False


@pytest.mark.parametrize("dropped", [False, True])
def test_collinearity_legacy_lists(data, dropped):
    data = data.assign(Xdup=data.X2, Zdup=data.Z1)
    fml = "Y ~ X2 + Xdup | f1 | X1 ~ Z1 + Zdup" if dropped else "Y ~ X2 | f1 | X1 ~ Z1"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        fit = pf.feols(fml, data)
    if dropped:
        assert any("collinear" in str(w.message) for w in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        for names, mask, expected_names, expected_mask in (
            (
                fit._collin_vars,
                fit._collin_index,
                ["Xdup"] if dropped else [],
                [False, True, False] if dropped else [],
            ),
            (
                fit._collin_vars_z,
                fit._collin_index_z,
                ["Zdup", "Xdup"] if dropped else [],
                [False, True, False, True] if dropped else [],
            ),
        ):
            assert isinstance(names, list)
            assert isinstance(mask, list)
            assert names == expected_names
            assert mask == expected_mask
        if not dropped:
            # Typed state retains full all-False masks; legacy access returns [].
            assert fit.collinearity.mask == (False, False)
            assert fit.collinearity_instruments.mask == (False, False)


@pytest.mark.parametrize("store_data,lean", [(True, True), (False, False)])
def test_retention_keeps_shims(data, fits, store_data, lean):
    """Retention cleanup drops instance state but never the class descriptors."""
    fit = pf.feols("Y ~ X1 | f1", data, weights="w", store_data=store_data, lean=lean)
    glm = pf.feglm(
        "Y_bin ~ X1 | f1", data, family="logit", store_data=store_data, lean=lean
    )
    assert "_X" in vars(Feols)
    assert "_u_hat" in vars(Feglm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        assert hasattr(fit, "_se")
        assert hasattr(fit, "_X") is not lean
        assert hasattr(fit, "_u_hat") is not lean
        assert hasattr(glm, "_u_hat") is not lean
        assert fit._N_rows == fit.sample_info.n_rows
        assert fit._has_weights
        assert fit._store_data is store_data
        assert fit._lean is lean
        assert fit._copy_data is fit.options.copy_data
        assert fit._demeaner is fit.options.demeaner
        assert fit._model_spec is fit.model.model_spec
    if lean:
        assert "within_data" not in vars(fit)
        assert "working_state" not in vars(glm)

    iv = pf.feols(
        "Y ~ X2 + [X1 ~ Z1] | f1",
        data,
        weights="w",
        vcov={"CRV1": "f1"},
        store_data=store_data,
        lean=lean,
    )
    quantile = pf.quantreg(
        "Y ~ X1 + X2",
        data,
        quantile=0.4,
        vcov="nid",
        store_data=store_data,
        lean=lean,
    )
    for name, model in (("Feiv", iv), ("Quantreg", quantile)):
        omitted = set(omitted_attributes(model.options.retention))
        assert "model_matrix" in omitted
        assert not hasattr(model, "model_matrix")
        # Array aliases need both the array's owner and any scaling state.
        # Quantile residual access reads _residuals directly, not resid().
        sources = {
            "_Y": ("within_data",),
            "_X": ("within_data",),
            "_Z": ("within_data",),
            "_u_hat": ("_residuals",),
            "_se": ("coeftable",),
            "_N_rows": ("sample_info",),
            "_fixef": ("model",),
            "_ssc_dict": ("options",),
            "_collin_vars": ("collinearity",),
        }
        if name == "Feiv":
            for alias in ("_Y", "_X", "_Z", "_u_hat"):
                sources[alias] += ("observation_weights",)
            sources.update(
                {
                    "_pi_hat": ("first_stage",),
                    "_X_hat": ("first_stage", "observation_weights"),
                    "_v_hat": ("first_stage", "observation_weights"),
                    "_non_exo_instruments": ("first_stage",),
                }
            )
        else:
            sources["_quantile"] = ("options",)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyfixestDeprecationWarning)
            for alias, dependencies in sources.items():
                for source in dependencies:
                    assert hasattr(model, source) == (source not in omitted), source
                missing = [
                    source for source in dependencies if not hasattr(model, source)
                ]
                if missing:
                    with pytest.raises(
                        AttributeError, match=rf"fit\.{alias}.*not available"
                    ) as excinfo:
                        getattr(model, alias)
                    message = str(excinfo.value)
                    assert any(source in message for source in missing), message
                    assert "deprecated; use `fit." in message
                else:
                    _assert_same(
                        getattr(model, alias),
                        getattr(fits[name], alias),
                        name=f"{name}.{alias}",
                    )
        # The first stage survives lean cleanup even though legacy fitted and
        # residual arrays cannot be scaled without observation_weights.
        if name == "Feiv":
            assert model.first_stage.fitted_values is not None
            assert model.first_stage.residuals is not None


def test_alpha_rejects_changed_categorical_layout():
    """Unused FE levels changed positional alpha; use labelled coefficients."""
    data = pd.DataFrame(
        {
            "y": np.arange(12, dtype=float),
            "x": np.tile([0.0, 1.0, 2.0], 4),
            "g": pd.Categorical(["a"] * 6 + ["b"] * 6, categories=["b", "unused", "a"]),
        }
    )
    fit = pf.feols("y ~ x | g", data)
    fit.fixef()
    with pytest.raises(AttributeError, match=r"layout changed.*fit.fixef\(\)"):
        fit._alpha  # noqa: B018


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_separated_glm_metadata_does_not_claim_preseparation_values(family):
    """Separated fits reject the old omitted-row set and removed level counts."""
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 0, 1, 1, 1],
            "x": [1, 2, 3, 4, 1, 2, 3, 4],
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    # Gaussian uses the same FE separation filter without binary perfect prediction.
    with pytest.warns(UserWarning, match="separation"):
        fit = pf.feglm("y ~ x | g", data, family=family, separation_check=["fe"])
    assert fit.sample_info.dropped_by_stage.separation == 4
    assert fit.sample_info.n_rows == 4
    assert fit.fixef_counts.n_levels_by_fe == (1,)
    assert fit.fixef_counts.n_fixef == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        # Non-Poisson GLMs reported the row count before separation, whereas
        # Poisson counted the retained sample and only FE with multiple levels.
        assert fit._N_rows == (8 if family == "gaussian" else 4)
        assert fit._n_fe == (1 if family == "gaussian" else 0)
    with pytest.raises(AttributeError, match="row positions before separation"):
        fit._na_index  # noqa: B018
    with pytest.raises(AttributeError, match=r"fit\.fixef_counts\.n_levels_by_fe"):
        fit._k_fe  # noqa: B018
