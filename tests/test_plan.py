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
# Method-specific model_kwargs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,must_have,must_not_have",
    [
        # feols: demeaner yes; iwls / quantreg / accelerate no
        ("feols", {"demeaner"}, {"tol", "maxiter", "offset", "accelerate", "quantile"}),
        # fepois: demeaner + iwls + separation_check + offset; no accelerate, no quantile
        (
            "fepois",
            {"demeaner", "tol", "maxiter", "separation_check", "offset"},
            {"accelerate", "quantile"},
        ),
        # feglm-logit: demeaner + iwls + separation_check + accelerate; no offset, no quantile
        (
            "feglm-logit",
            {"demeaner", "tol", "maxiter", "separation_check", "accelerate"},
            {"offset", "quantile"},
        ),
        # quantreg: quantile knobs only; no demeaner, no iwls, no separation_check
        (
            "quantreg",
            {"quantile", "method", "quantile_tol", "quantile_maxiter", "seed"},
            {"demeaner", "tol", "maxiter", "separation_check", "offset", "accelerate"},
        ),
    ],
)
def test_model_kwargs_filtered_by_method(method, must_have, must_not_have):
    """`expand_specs` only threads kwargs the model class consumes."""
    data = pf.get_data()
    # quantreg needs a quantile value
    overrides = {"quantile": 0.5} if method.startswith("quantreg") else {}
    cfg = _config(method, "Y ~ X1", data, **overrides)
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
    kwargs = specs[0].model_kwargs
    for key in must_have:
        assert key in kwargs, f"{method} should have kwarg {key!r}"
    for key in must_not_have:
        assert key not in kwargs, f"{method} should not have kwarg {key!r}"


def test_cache_dicts_are_not_in_spec_kwargs():
    """`lookup_demeaned_data` and `lookup_preconditioner` are runner-injected."""
    data = pf.get_data()
    cfg = _config("feols", "Y ~ X1 | f1", data)
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
    assert "lookup_demeaned_data" not in specs[0].model_kwargs
    assert "lookup_preconditioner" not in specs[0].model_kwargs


# ---------------------------------------------------------------------------
# End-to-end smoke: planner output matches the public API
# ---------------------------------------------------------------------------


def test_public_feols_matches_legacy_behavior():
    """Sanity check: the planner doesn't change end-to-end results."""
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1 + f2", data)
    # If the planner regressed anything, coefficients would shift.
    assert abs(fit.coef().iloc[0] - (-0.9240461507764969)) < 1e-10
