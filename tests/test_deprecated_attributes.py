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

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.errors import PyfixestDeprecationWarning
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
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data, weights="w", vcov={"CRV1": "f1"})
    iv.IV_Diag()
    glm = pf.feglm("Y_bin ~ X1 + X2 | f1", data, family="logit", vcov="hetero")
    glm.fixef()
    poisson = pf.fepois("Y_count ~ X1 + X2 | f1", data, weights="w", vcov="hetero")
    poisson.fixef()
    quantile = pf.quantreg("Y ~ X1 + X2", data, quantile=0.4, vcov="nid")
    return {
        "Feols": ols,
        "Feiv": iv,
        "Feglm": glm,
        "Fepois": poisson,
        "Quantreg": quantile,
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
    if name in EVENT_STUDY_ROWS:
        return event_study
    if name == "_sample_split_value":
        return split_fit
    return fits[class_name]


def _assert_same(value, expected) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(value, expected)
    else:
        assert value == expected or value is expected


def test_entries_install_on_model_classes():
    """Every entry installs a descriptor on the model class it names."""
    for attribute in DEPRECATED_ATTRIBUTES:
        for class_name in attribute.classes:
            assert class_name in MODEL_CLASSES, attribute.name
            assert attribute.name in vars(MODEL_CLASSES[class_name]), attribute.name
    names = [attribute.name for attribute in DEPRECATED_ATTRIBUTES]
    assert len(names) == len(set(names))


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
        _assert_same(value, replacement)


@pytest.mark.parametrize(("attribute", "class_name", "entry"), REMOVED)
def test_removed_attribute_names_replacement(
    attribute, class_name, entry, fits, event_study, split_fit
):
    fit = _fit_for(attribute.name, class_name, fits, event_study, split_fit)
    with pytest.raises(AttributeError, match="Sorry") as excinfo:
        getattr(fit, attribute.name)
    assert f"`{entry.replacement}`" in str(excinfo.value)
    assert not hasattr(fit, attribute.name)


def test_unpublished_state_stays_missing(data):
    """Deprecated names raise `AttributeError` until their state is published."""
    fit = pf.feols("Y ~ X1 | f1", data)
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1]", data)
    full_sample = pf.feols("Y ~ X1", data, fsplit="f2").to_list()[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("_alpha", "_sumFE", "_sample_split_value", "_yname"):
            assert not hasattr(fit, name), name
        assert not hasattr(full_sample, "_sample_split_value")
        with pytest.raises(AttributeError, match=r"IV_Diag\(\)"):
            iv._eff_F  # noqa: B018
    with pytest.raises(AttributeError, match=re.escape("fit.coeftable.se")):
        fit._se = np.zeros(1)


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
        # 0.60: the weighted first stage splits the endogenous regressor
        np.testing.assert_allclose(
            fit._X_hat + fit._v_hat, sqrt_w * within.endogenous.flatten()
        )
        assert fit._non_exo_instruments == ["Z1"]
        assert fit._p_iv == 1


def test_glm_rebuilds_working_weighted_arrays(fits):
    """0.60 GLMs stored the final IRLS arrays weighted by sqrt(working weights)."""
    for name in ("Feglm", "Fepois"):
        fit = fits[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyfixestDeprecationWarning)
            X, Y, u_hat = fit._X, fit._Y.flatten(), fit._u_hat
            np.testing.assert_allclose(u_hat, Y - X @ fit._beta_hat, atol=1e-10)
            np.testing.assert_allclose(fit._scores, X * u_hat[:, None], atol=1e-10)
            np.testing.assert_allclose(fit._hessian, X.T @ X)
            np.testing.assert_allclose(
                fit._weights.flatten(), fit.working_state.working_weights
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        assert fits["Fepois"]._Y.shape == (fits["Fepois"].sample_info.n_rows, 1)
        assert fits["Feglm"]._Y.ndim == 1
        # weighted Poisson fits had no null model in 0.60
        assert fits["Fepois"]._loglik_null is None


def test_cluster_counts_keep_min_padding(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        oneway = pf.feols("Y ~ X1", data, vcov={"CRV1": "f1"})
        G = oneway.variance_covariance.G[0]
        assert oneway._G == [G, G, G]
        assert oneway._ssc.shape == (1, 1)
        conventional = pf.feols(
            "Y ~ X1", data, vcov={"CRV1": "f1+f2"}, ssc=pf.ssc(G_df="conventional")
        )
        assert list(conventional.variance_covariance.G) == conventional._G
        assert len(conventional._G) == 3
        assert pf.feols("Y ~ X1", data, vcov="hetero")._G == []


@pytest.mark.parametrize("store_data,lean", [(True, True), (False, False)])
def test_retention_keeps_shims(data, store_data, lean):
    """Retention cleanup drops instance state but never the class descriptors."""
    fit = pf.feols("Y ~ X1 | f1", data, weights="w", store_data=store_data, lean=lean)
    glm = pf.feglm("Y_bin ~ X1 | f1", data, family="logit", lean=lean)
    assert "_X" in vars(Feols)
    assert "_u_hat" in vars(Feglm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        assert hasattr(fit, "_se")
        assert hasattr(fit, "_X") is not lean
        assert hasattr(glm, "_u_hat") is not lean
    if lean:
        assert "within_data" not in vars(fit)
        assert "working_state" not in vars(glm)
