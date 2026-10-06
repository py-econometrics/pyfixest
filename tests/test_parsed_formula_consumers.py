from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.internals.model_state import VcovSpec
from pyfixest.estimation.post_estimation.ritest import _resample
from pyfixest.estimation.refit import refit


def _fail_on_reparse(*args, **kwargs):
    raise AssertionError("Internal consumers must reuse the parsed formula.")


@pytest.mark.parametrize("statistic", ["randomization-c", "randomization-t"])
@pytest.mark.parametrize("estimator", [pf.feols, pf.fepois])
def test_ritest_uses_parsed_formula(estimator, statistic, monkeypatch):
    """No-intercept RI retains controls whose names contain the treatment name."""
    data = pf.get_data(N=300, model="Fepois").dropna()
    data["D"] = (data.X1 > 0).astype(float)
    data["D_control"] = data.X2
    # Neither a pre-existing resampled column nor a substring in a control
    # should be overwritten or substituted in the formula.
    data["D_resampled"] = data.X1
    fml = "Y ~ D + D_control + {D * D_control} - 1"
    fit = estimator(fml, data=data)
    original = fit._data.copy(deep=True)
    reps = 5
    rng = np.random.default_rng(31)
    expected = []
    for _ in range(reps):
        resampled = original.copy(deep=False)
        resampled["D"] = _resample(
            resampvar_arr=original.D.to_numpy(), rng=rng
        ).flatten()
        reference = estimator(
            fml,
            data=resampled,
            vcov="iid" if statistic == "randomization-c" else "hetero",
        )
        values = (
            reference.coef() if statistic == "randomization-c" else reference.tstat()
        )
        expected.append(values["D"])

    monkeypatch.setattr(
        Formula, "formula", property(lambda self: "descriptive text, not a formula")
    )
    assert fit.model.formula == fit._fml == "descriptive text, not a formula"
    monkeypatch.setattr(Formula, "parse", _fail_on_reparse)
    fit.ritest(
        "D",
        reps=reps,
        type=statistic,
        choose_algorithm="slow",
        rng=np.random.default_rng(31),
        store_ritest_statistics=True,
    )
    np.testing.assert_allclose(fit.ritest_statistics.statistics, expected, rtol=1e-10)
    pd.testing.assert_frame_equal(fit._data, original)


@pytest.mark.parametrize(
    "fml", ["Y ~ 0 | f1", "Y ~ i(f1) - 1", "Y ~ X2 - 1 + [X1 ~ Z1]"]
)
def test_refit_retains_parsed_stages(fml, monkeypatch):
    data = pf.get_data(N=300).dropna()
    fit = pf.feols(fml, data=data)
    monkeypatch.setattr(Formula, "parse", _fail_on_reparse)
    replay = refit(
        fit,
        data=fit._data,
        vcov=VcovSpec(vcov_type="iid", vcov_type_detail="iid"),
    )
    pd.testing.assert_series_equal(replay.coef(), fit.coef())


@pytest.mark.parametrize("output", ["numpy", "sparse"])
@pytest.mark.parametrize("fixed_effects", ["f1", "`my fe`", "f1:f2"])
def test_one_hot_uses_parsed_terms(output, fixed_effects, monkeypatch):
    data = pf.get_data(N=300).dropna()
    data["my fe"] = data.f1
    data["group"] = data.groupby(["f1", "f2"]).ngroup()
    data["product"] = data.X1 * data.X2
    fit = pf.feols(
        f"Y ~ {{X1 * X2}} + X2 - 1 | {fixed_effects}",
        data=data,
        fixef_rm="none",
    )
    reference = pf.feols(
        "Y ~ product + X2 + C(`my fe`)"
        if fixed_effects != "f1:f2"
        else "Y ~ product + X2 + C(group)",
        data=data,
    )
    monkeypatch.setattr(
        Formula, "formula", property(lambda self: "descriptive text, not a formula")
    )
    monkeypatch.setattr(Formula, "parse", _fail_on_reparse)
    y, x, names = fit._model_matrix_one_hot(output=output)
    x = x.toarray() if output == "sparse" else x
    assert "X1 * X2" in names
    label = fixed_effects.replace("`", "")
    assert any(label in name for name in names)
    assert all("__fixed_effect__" not in name for name in names)
    np.testing.assert_allclose(y, reference.within_data.response.flatten())
    # Dummy names can differ in their quoting, but their values and order agree.
    np.testing.assert_allclose(x, reference.within_data.design)


@pytest.mark.parametrize("lhs", ["`my outcome`", "{Y * 2}"])
def test_separation_uses_parsed_formula_and_materialized_response(lhs, monkeypatch):
    data = pd.DataFrame(
        {
            "Y": [0, 0, 0, 1, 2, 3, 1, 2],
            "X": [1, 2, 1, 2, 3, 4, 2, 3],
            "fe": ["a", "a", "b", "b", "b", "b", "b", "b"],
            "U": np.arange(8),
            "U_separationTmp": np.arange(8),
            "omega": np.ones(8),
            "Uhat": np.ones(8),
        }
    )
    data["my outcome"] = data.Y
    original = data.copy(deep=True)
    expected = pf.fepois("Y ~ X - 1 | fe", data=data, separation_check=["ir"])
    parse = Formula.parse
    calls = []

    def parse_once(formula):
        calls.append(formula)
        assert len(calls) == 1, "Separation must not parse an auxiliary string."
        return parse(formula)

    monkeypatch.setattr(Formula, "parse", parse_once)
    fit = pf.fepois(f"{lhs} ~ X - 1 | fe", data=data, separation_check=["ir"])
    assert fit.sample_info.n_obs == expected.sample_info.n_obs == 6
    # Scaling a Poisson response shifts the FE intercepts, leaving its slope.
    np.testing.assert_allclose(fit.coef(), expected.coef(), rtol=1e-6, atol=1e-8)
    pd.testing.assert_frame_equal(data, original)
