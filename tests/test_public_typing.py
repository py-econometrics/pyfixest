"""Tests for public aliases and statically declared result methods."""

from __future__ import annotations

import inspect
from typing import get_args

import pyfixest as pf
import pyfixest.typing as pft
from pyfixest.estimation.FixestMulti_ import FixestMulti
from pyfixest.estimation.models.feiv_ import Feiv


def test_public_literal_aliases_match_supported_options() -> None:
    assert set(get_args(pft.RegressionVcovType)) == {
        "iid",
        "hetero",
        "HC1",
        "HC2",
        "HC3",
        "NW",
        "DK",
    }
    assert set(get_args(pft.QuantregVcovType)) == {
        "iid",
        "hetero",
        "HC1",
        "HC2",
        "HC3",
        "nid",
    }
    assert set(get_args(pft.PlotBackend)) == {"matplotlib", "lets_plot"}
    assert set(get_args(pft.EventStudyEstimator)) == {
        "twfe",
        "did2s",
        "saturated",
    }
    assert set(get_args(pft.WeightsType)) == {"aweights", "fweights"}
    assert set(get_args(pft.GlmFamily)) == {
        "gaussian",
        "logit",
        "probit",
        "poisson",
    }


def test_report_methods_are_declared_on_result_types() -> None:
    for cls in (pf.estimation.Feols, FixestMulti):
        for method_name in ("summary", "etable", "coefplot", "iplot"):
            assert inspect.isfunction(inspect.getattr_static(cls, method_name))

    data = pf.get_data()
    fit = pf.feols("Y ~ X1", data=data)
    fit_multi = pf.feols("Y + Y2 ~ X1", data=data)

    assert "summary" not in vars(fit)
    assert "summary" not in vars(fit_multi)
    assert inspect.ismethod(fit.summary)
    assert inspect.ismethod(fit_multi.summary)


def test_feiv_diagnostic_properties_retain_private_state() -> None:
    fit = pf.feols("Y ~ 1 | f1 | X1 ~ Z1", data=pf.get_data(), vcov="hetero")

    assert isinstance(fit, Feiv)
    assert fit.first_stage_model is fit._model_1st_stage
    assert fit.first_stage_f_statistic == float(fit._f_stat_1st_stage)

    fit.IV_Diag()
    assert fit.effective_f_statistic == float(fit._eff_F)
