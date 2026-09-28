"""Published model descriptions across estimators, splits, and retention."""

from __future__ import annotations

import pytest

import pyfixest as pf
from pyfixest.estimation.internals.families import NORMAL_DIST, T_DIST


@pytest.mark.parametrize(
    "retention", [{}, {"store_data": False}, {"lean": True}], ids=str
)
def test_matrix_time_fields_survive_retention(retention):
    fit = pf.feols("Y ~ X1 + i(f2) | f1 + f3", pf.get_data(), **retention)

    assert fit.model.depvar == "Y"
    assert fit.model.fixed_effects == ("f1", "f3")
    assert fit.model.fixef == "f1+f3"
    assert fit.model.interacted_covariates == tuple(
        name for name in fit._coefnames if name.startswith("f2::")
    )
    assert fit.model.model_spec is not None


def test_iv_glm_and_quantile_descriptions_record_the_fitted_method():
    data = pf.get_data()
    iv = pf.feols("Y ~ X2 + [X1 ~ Z1] | f1", data)
    assert iv.model.method == "feols"
    assert iv.model.is_iv is True
    assert iv.model.inference_dist is T_DIST

    poisson = pf.fepois("Y ~ X1", pf.get_data(model="Fepois"))
    assert poisson.model.method == "fepois"
    assert poisson.model.inference_dist is NORMAL_DIST

    binary_data = data.copy()
    binary_data["Y_bin"] = (binary_data["Y"] > 0).astype(int)
    logit = pf.feglm("Y_bin ~ X1", binary_data, family="logit")
    assert logit.model.method == "feglm-logit"
    assert logit.model.inference_dist is NORMAL_DIST

    gaussian = pf.feglm("Y ~ X1", data, family="gaussian")
    assert gaussian.model.method == "feglm-gaussian"
    assert gaussian.model.inference_dist is T_DIST

    quantile = pf.quantreg("Y ~ X1", data, quantile=0.25, method="pfn")
    assert quantile.model.method == "quantreg_pfn"
    assert quantile.model.model_name == "Y ~ 1 + X1 (q = 0.25)"
    assert quantile._model_name_plot == quantile.model.model_name
    assert quantile.model.inference_dist is T_DIST


def test_split_description_names_the_sample():
    data = pf.get_data()
    unsplit = pf.feols("Y ~ X1", data)
    assert unsplit.model.sample_split_var is None
    assert unsplit.model.sample_split_value is None
    assert unsplit.model.model_name == "Y ~ 1 + X1"

    fits = pf.feols("Y ~ X1", data, fsplit="f1").to_list()
    levels = data["f1"].dropna().drop_duplicates().sort_values().tolist()
    assert [fit.model.sample_split_value for fit in fits[1:]] == levels
    for fit in fits:
        assert fit.model.sample_split_var == "f1"
        value = fit.model.sample_split_value
        assert fit.model.model_name == f"Y ~ 1 + X1 (Sample: f1 = {value})"
    assert repr(fits[0].model.sample_split_value) == "all"
