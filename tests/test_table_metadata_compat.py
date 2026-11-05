"""Compatibility for metadata used by released table integrations."""

import warnings

import pytest

import pyfixest as pf
from pyfixest.errors import PyfixestDeprecationWarning


@pytest.fixture
def data():
    return pf.get_data(N=300, seed=2026).dropna().reset_index(drop=True)


def test_table_metadata_and_split_are_silent(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyfixestDeprecationWarning)
        fit = pf.feols("Y ~ X1 | f1", data, vcov={"CRV1": "f1"})
        table = pf.etable(fit, type="df")
        assert table.loc[("stats", "Observations")].iloc[0] == str(
            fit.sample_info.n_obs
        )
        assert table.loc[("stats", "R²")].iloc[0] == f"{fit.fitstat.r2:.3f}"
        assert table.loc[("fe", "f1")].iloc[0] == "x"
        assert [min(fit.variance_covariance.G)] * 3 == fit._G
        assert fit._clustervar == ["f1"]
        full = pf.feols("Y ~ X1", data, fsplit="f2").to_list()[0]
        assert full._sample_split_value == "all"
        assert full._sample_split_var == "f2"


@pytest.mark.parametrize("lean,store_data", [(True, True), (False, False)])
def test_table_metadata_survives_retention(data, lean, store_data):
    fit = pf.feols("Y ~ X1 | f1", data, lean=lean, store_data=store_data)
    assert fit.sample_info.n_obs == fit._N
    assert fit._r2 == fit.fitstat.r2
    assert fit._fixef == fit.model.fixef


def test_aliases_are_read_only(data):
    fit = pf.feols("Y ~ X1", data)
    with pytest.raises(AttributeError, match="read-only"):
        fit._N = 1
    with pytest.raises(AttributeError, match="read-only"):
        del fit._N
