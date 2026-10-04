"""Published model descriptions across estimators, splits, and retention."""

from __future__ import annotations

import numpy as np
import pytest

import pyfixest as pf
from pyfixest.estimation.internals.families import NORMAL_DIST, T_DIST


@pytest.mark.parametrize(
    "retention", [{}, {"store_data": False}, {"lean": True}], ids=str
)
def test_matrix_time_fields_survive_retention(retention):
    data = pf.get_data().rename(columns={"X1": "a::b"})
    fit = pf.feols("Y ~ Q('a::b') + i(f2) | f1 + f3", data, **retention)

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
    assert unsplit.model.sample_split is None
    assert unsplit.model.model_name == "Y ~ 1 + X1"

    fits = pf.feols("Y ~ X1", data, fsplit="f1").to_list()
    levels = data["f1"].dropna().drop_duplicates().sort_values().tolist()
    assert [fit.model.sample_split.value for fit in fits] == [None, *levels]
    for fit, label in zip(fits, ["all", *levels], strict=True):
        assert fit.model.sample_split.var == "f1"
        assert fit.model.model_name == f"Y ~ 1 + X1 (Sample: f1 = {label})"
        assert (fit.tidy()["Sample"] == label).all()


def test_summary_names_a_split_group_called_all(capsys):
    data = pf.get_data()
    data["group"] = np.where(data["f1"] < 10, "all", "rest")
    group_all, _ = pf.feols("Y ~ X1", data, split="group").to_list()
    assert group_all.model.sample_split.value == "all"

    group_all.summary()
    assert "sample: group = all" in capsys.readouterr().out
