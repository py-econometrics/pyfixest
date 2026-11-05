"""Check the deprecation shims that keep 0.60 fitted-model attributes readable.

Every row of `DEPRECATED_ATTRIBUTES` is exercised on a fit of each class it
names. Metadata values are compared with pyfixest 0.60.0 in
`test_release_contract.py`; these tests cover access, warnings, defaults, and
retention without changing the estimator's array representations.
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
from pyfixest.estimation.api.utils import _ALL_SAMPLE
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

# Rows whose 0.60 value is a rebuilt form of the replacement (a list,
# a `None` default, or padding) rather than the
# replacement itself. The release contract compares them with 0.60.0.
REBUILT = {
    "get_performance",
    "_sample_split_var",
    "_N_rows",
    "_n_fe",
    "_fixef",
    "_icovars",
    "_ssc_dict",
    "_clustervar",
    "_G",
    "deviance",
    "_loglik_null",
    "_pseudo_r2",
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
            if split is None or split.value is None:
                assert fit._sample_split_value is _ALL_SAMPLE
            else:
                assert fit._sample_split_value == split.value
        fit = fits["Feols"]
        table = pf.etable(fit, type="df")
        assert table.loc[("stats", "Observations")].iloc[0] == str(
            fit.sample_info.n_obs
        )
        assert table.loc[("stats", "R²")].iloc[0] == f"{fit.fitstat.r2:.3f}"
        assert table.loc[("fe", "f1")].iloc[0] == "x"


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


@pytest.mark.parametrize("family", ["logit", "poisson"])
def test_separated_glm_metadata_does_not_claim_preseparation_values(family):
    """Separated fits reject the old omitted-row set and removed level counts."""
    data = pd.DataFrame(
        {
            "y": [0, 0, 0, 0, 0, 1, 1, 0],
            "x": [1, 2, 3, 4, 1, 2, 3, 4],
            "g": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    # The retained logit group has both outcomes and no perfect prediction.
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
        assert fit._N_rows == (8 if family == "logit" else 4)
        assert fit._n_fe == (1 if family == "logit" else 0)
    with pytest.raises(AttributeError, match="row positions before separation"):
        fit._na_index  # noqa: B018
    with pytest.raises(AttributeError, match=r"fit\.fixef_counts\.n_levels_by_fe"):
        fit._k_fe  # noqa: B018


@pytest.mark.parametrize("store_data,lean", [(True, True), (False, False)])
def test_metadata_aliases_survive_retention(data, store_data, lean):
    fit = pf.feols("Y ~ X1 | f1", data, store_data=store_data, lean=lean)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyfixestDeprecationWarning)
        np.testing.assert_array_equal(fit._se, fit.coeftable.se)
        assert fit.sample_info.n_obs == fit._N
        assert fit._store_data is store_data
        assert fit._lean is lean
    assert "_se" not in vars(fit)
    assert hasattr(fit, "within_data") is not lean
