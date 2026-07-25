"""Unit tests for the estimation planner (`pyfixest.estimation.plan_`)."""

from __future__ import annotations

import pytest

import pyfixest as pf
from pyfixest.estimation.api.utils import _ALL_SAMPLE
from pyfixest.estimation.config import EstimationConfig
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.models.fegaussian_ import Fegaussian
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.felogit_ import Felogit
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import (
    MODEL_REGISTRY,
    _resolve_model_class,
    build_all_splits,
    expand_specs,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg


def _config(method: str, fml: str, data, **overrides) -> EstimationConfig:
    """Minimal config builder for planner tests."""
    base = dict(
        method=method,
        data=data,
        fml=fml,
        ssc_dict={},
        context={},
    )
    base.update(overrides)
    return EstimationConfig(**base)


def _parse(fml: str):
    return Formula.parse_to_dict(fml)


def _is_iv(formula_dict) -> bool:
    return any(f.first_stage is not None for fs in formula_dict.values() for f in fs)


# ---------------------------------------------------------------------------
# Model registry / dispatch
# ---------------------------------------------------------------------------


def test_registry_covers_every_supported_method():
    expected = {
        "feols",
        "fepois",
        "feglm-logit",
        "feglm-probit",
        "feglm-gaussian",
        "quantreg",
        "quantreg_multi",
    }
    assert set(MODEL_REGISTRY.keys()) == expected


@pytest.mark.parametrize(
    "method,is_iv,expected_cls",
    [
        ("feols", False, Feols),
        ("feols", True, Feiv),
        ("fepois", False, Fepois),
        ("feglm-logit", False, Felogit),
        ("feglm-gaussian", False, Fegaussian),
        ("quantreg", False, Quantreg),
    ],
)
def test_resolve_model_class(method, is_iv, expected_cls):
    assert _resolve_model_class(method, is_iv) is expected_cls


def test_iv_only_promotes_feols():
    """is_iv=True for any non-feols method falls back to the registry entry."""
    assert _resolve_model_class("fepois", is_iv=True) is Fepois


# ---------------------------------------------------------------------------
# Split enumeration
# ---------------------------------------------------------------------------


def test_build_all_splits_full_only():
    data = pf.get_data()
    splits = build_all_splits(run_full=True, run_split=False, splitvar=None, data=data)
    assert splits == [_ALL_SAMPLE]


def test_build_all_splits_split_only():
    data = pf.get_data()
    splits = build_all_splits(run_full=False, run_split=True, splitvar="f1", data=data)
    expected = sorted(data["f1"].dropna().unique().tolist())
    assert splits == expected


def test_build_all_splits_full_plus_split_puts_full_first():
    data = pf.get_data()
    splits = build_all_splits(run_full=True, run_split=True, splitvar="f1", data=data)
    assert splits[0] is _ALL_SAMPLE
    assert splits[1:] == sorted(data["f1"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# expand_specs: spec count & ordering
# ---------------------------------------------------------------------------


def test_single_formula_emits_one_spec():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X1 + X2 | f1", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[_ALL_SAMPLE],
        is_iv=False,
        splitvar=None,
        captured_context={},
    )
    assert len(specs) == 1
    assert specs[0].method == "feols"
    assert specs[0].model_cls is Feols
    assert specs[0].cache_key == (_ALL_SAMPLE, "f1")


def test_csw_emits_one_spec_per_fixef_step():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X1 | csw(f1, f2)", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[_ALL_SAMPLE],
        is_iv=False,
        splitvar=None,
        captured_context={},
    )
    # csw(f1, f2) → two fixef keys: "f1" then "f1+f2"
    assert len(specs) == 2
    assert specs[0].fixef_key != specs[1].fixef_key


def test_cache_keys_are_contiguous_blocks():
    """Cache blocks form contiguous runs in the spec list.

    This is the invariant the runner relies on to drop the demean /
    preconditioner cache without re-allocating per spec.
    """
    data = pf.get_data()
    cfg = _config("feols", "Y + Y2 ~ X1 | csw(f1, f2)", data)
    fd = _parse(cfg.fml)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[_ALL_SAMPLE],
        is_iv=False,
        splitvar=None,
        captured_context={},
    )
    seen: list = []
    for spec in specs:
        if not seen or spec.cache_key != seen[-1]:
            seen.append(spec.cache_key)
    # Each cache_key should appear in `seen` exactly once if blocks
    # are contiguous — i.e. once the runner has left a block it
    # never comes back.
    assert len(seen) == len(set(seen))


def test_split_expansion_walks_full_then_each_split_value():
    data = pf.get_data()
    cfg = _config(
        "feols",
        "Y ~ X1 | f1",
        data,
        fsplit="f2",
    )
    fd = _parse(cfg.fml)
    splits = build_all_splits(run_full=True, run_split=True, splitvar="f2", data=data)
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=splits,
        is_iv=False,
        splitvar="f2",
        captured_context={},
    )
    assert len(specs) == len(splits)
    assert [s.sample_split_value for s in specs] == splits


def test_iv_formula_resolves_each_spec_to_feiv():
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X2 | f1 | X1 ~ Z1", data)
    fd = _parse(cfg.fml)
    is_iv = _is_iv(fd)
    assert is_iv
    specs = expand_specs(
        config=cfg,
        formula_dict=fd,
        data=data,
        splits=[_ALL_SAMPLE],
        is_iv=is_iv,
        splitvar=None,
        captured_context={},
    )
    assert all(s.model_cls is Feiv for s in specs)


# ---------------------------------------------------------------------------
# Method-specific constructor extras
# ---------------------------------------------------------------------------


def _first_spec(method: str, fml: str = "Y ~ X1", **overrides):
    data = pf.get_data()
    if method.startswith("quantreg"):
        overrides.setdefault("quantile", 0.5)
    cfg = _config(method, fml, data, **overrides)
    return expand_specs(
        config=cfg,
        formula_dict=_parse(cfg.fml),
        data=data,
        splits=[_ALL_SAMPLE],
        is_iv=False,
        splitvar=None,
        captured_context={},
    )[0]


@pytest.mark.parametrize(
    "method,must_have,must_not_have",
    [
        # feols takes no extras at all
        ("feols", set(), {"tol", "maxiter", "offset", "accelerate", "quantile"}),
        # fepois: iwls + separation_check + offset; no accelerate, no quantile
        (
            "fepois",
            {"tol", "maxiter", "separation_check", "offset"},
            {"accelerate", "quantile"},
        ),
        # feglm-logit: iwls + separation_check + accelerate; no offset, no quantile
        (
            "feglm-logit",
            {"tol", "maxiter", "separation_check", "accelerate"},
            {"offset", "quantile"},
        ),
        # quantreg: quantile knobs only
        (
            "quantreg",
            {"quantile", "method", "quantile_tol", "quantile_maxiter", "seed"},
            {"tol", "maxiter", "separation_check", "offset", "accelerate"},
        ),
    ],
)
def test_model_extras_filtered_by_method(method, must_have, must_not_have):
    """`expand_specs` only threads the extras the model class consumes."""
    extras = _first_spec(method).extras
    for key in must_have:
        assert key in extras, f"{method} should have extra {key!r}"
    for key in must_not_have:
        assert key not in extras, f"{method} should not have extra {key!r}"


@pytest.mark.parametrize(
    "method,threads_demeaner", [("feols", True), ("fepois", True), ("quantreg", False)]
)
def test_demeaner_threaded_only_when_consumed(method, threads_demeaner):
    """The demeaner reaches `ModelInit` only for methods that declare the need."""
    demeaner = pf.MapDemeaner()
    spec = _first_spec(method, demeaner=demeaner)
    assert (spec.init.demeaner is demeaner) == threads_demeaner


def test_cache_fields_are_runner_injected():
    """`lookup_demeaned_data` and `lookup_preconditioner` are left unset by the planner."""
    spec = _first_spec("feols", "Y ~ X1 | f1")
    assert spec.init.lookup_demeaned_data == {}
    assert spec.init.lookup_preconditioner is None


# ---------------------------------------------------------------------------
# End-to-end smoke: planner output matches the public API
# ---------------------------------------------------------------------------


def test_public_feols_matches_legacy_behavior():
    """Sanity check: the planner doesn't change end-to-end results."""
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    # If the planner regressed anything, coefficients would shift.
    assert abs(fit.coef().iloc[0] - (-0.9240461507764969)) < 1e-10
