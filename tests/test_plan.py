"""Unit tests for the estimation plan (ModelSpec / expand_specs / fit_one).

These tests exercise the pure planning layer without fitting models
(except for one cheap end-to-end smoke test of ``fit_one``).
"""

import numpy as np
import pandas as pd
import pytest

import pyfixest as pf
from pyfixest.demeaners import MapDemeaner
from pyfixest.estimation.api.utils import _ALL_SAMPLE
from pyfixest.estimation.formula.parse import Formula
from pyfixest.estimation.models.feiv_ import Feiv
from pyfixest.estimation.models.feols_ import Feols
from pyfixest.estimation.models.fepois_ import Fepois
from pyfixest.estimation.plan_ import (
    MODEL_REGISTRY,
    expand_specs,
    fit_one,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg
from pyfixest.utils.utils import ssc as ssc_func


def _expand(fml, *, method="feols", is_iv=False, data=None, **overrides):
    "Call expand_specs with sensible defaults for unit testing."
    if data is None:
        data = pd.DataFrame({"f1": [1, 1, 2, 2]})
    defaults = dict(
        formula_dict=Formula.parse_to_dict(fml),
        method=method,
        is_iv=is_iv,
        data=data,
        splitvar=None,
        run_full=True,
        run_split=False,
        ssc_dict=ssc_func(),
        drop_singletons=False,
        drop_intercept=False,
        weights=None,
        weights_type="aweights",
        solver="np.linalg.solve",
        collin_tol=1e-09,
        store_data=True,
        copy_data=True,
        lean=False,
        context={},
        demeaner=MapDemeaner(),
    )
    defaults.update(overrides)
    return expand_specs(**defaults)


def test_single_model_single_spec():
    specs = _expand("Y ~ X1 | f1")

    assert len(specs) == 1
    spec = specs[0]
    assert spec.model_cls is Feols
    assert spec.method == "feols"
    assert spec.sample_split_value is _ALL_SAMPLE
    assert spec.cache_key == (_ALL_SAMPLE, spec.fixef_key)
    # shared kwargs present, method-specific kwargs absent
    assert "demeaner" in spec.model_kwargs
    assert "offset" not in spec.model_kwargs
    assert "accelerate" not in spec.model_kwargs
    assert "quantile" not in spec.model_kwargs
    # data and the demean cache are injected by the runner, not the spec
    assert "data" not in spec.model_kwargs
    assert "lookup_demeaned_data" not in spec.model_kwargs


def test_multiple_estimation_expansion_and_cache_blocks():
    # 2 dependent variables x 2 stepwise fixef keys -> 4 models
    specs = _expand("Y + Y2 ~ X1 | sw(f1, f2)")

    assert len(specs) == 4

    # models with the same fixef key share one demean-cache block ...
    cache_keys = [spec.cache_key for spec in specs]
    assert len(set(cache_keys)) == 2

    # ... and the blocks are contiguous: once a cache key is left,
    # it never recurs (this is what allows the runner to drop a block's
    # cache as soon as the next block starts)
    seen: list = []
    for key in cache_keys:
        if key not in seen:
            seen.append(key)
        else:
            assert key == seen[-1], "cache blocks must be contiguous"


def test_split_expansion():
    data = pd.DataFrame({"split_var": ["b", "a", "a", "b"]})

    specs = _expand(
        "Y ~ X1",
        data=data,
        splitvar="split_var",
        run_full=True,
        run_split=True,
    )

    # full sample + one model per (sorted) split level
    assert len(specs) == 3
    assert specs[0].sample_split_value is _ALL_SAMPLE
    assert [s.sample_split_value for s in specs[1:]] == ["a", "b"]
    assert all(s.sample_split_var == "split_var" for s in specs)
    assert all(s.model_kwargs["sample_split_var"] == "split_var" for s in specs)


def test_iv_resolves_to_feiv():
    specs_ols = _expand("Y ~ X1 | f1", is_iv=False)
    specs_iv = _expand("Y ~ 1 | f1 | X1 ~ Z1", is_iv=True)

    assert specs_ols[0].model_cls is Feols
    assert specs_iv[0].model_cls is Feiv


@pytest.mark.parametrize(
    ("method", "model_cls", "present", "absent"),
    [
        ("feols", Feols, ["demeaner"], ["offset", "tol", "accelerate", "quantile"]),
        (
            "fepois",
            Fepois,
            ["demeaner", "offset", "tol", "maxiter", "separation_check"],
            ["accelerate", "quantile"],
        ),
        (
            "feglm-logit",
            MODEL_REGISTRY["feglm-logit"].model_cls,
            ["demeaner", "tol", "maxiter", "accelerate"],
            ["offset", "quantile"],
        ),
        (
            "quantreg",
            Quantreg,
            ["quantile", "method", "quantile_tol", "seed"],
            ["demeaner", "offset", "accelerate"],
        ),
    ],
)
def test_method_specific_kwargs(method, model_cls, present, absent):
    specs = _expand("Y ~ X1 | f1", method=method, quantile=0.5)

    spec = specs[0]
    assert spec.model_cls is model_cls
    for key in present:
        assert key in spec.model_kwargs, f"{key} missing for {method}"
    for key in absent:
        assert key not in spec.model_kwargs, f"{key} unexpected for {method}"


def test_fit_one_smoke():
    "fit_one on a spec must reproduce the api result exactly."
    rng = np.random.default_rng(42)
    N = 200
    data = pd.DataFrame(
        {
            "Y": rng.normal(size=N),
            "X1": rng.normal(size=N),
            "f1": rng.integers(0, 5, N),
        }
    )

    # match the feols() api defaults so results are exactly equal
    specs = _expand("Y ~ X1 | f1", data=data, solver="scipy.linalg.solve")
    fit = fit_one(
        specs[0],
        data=data,
        lookup_demeaned_data={},
        vcov="iid",
        vcov_kwargs=None,
    )

    fit_api = pf.feols("Y ~ X1 | f1", data=data, vcov="iid")

    assert isinstance(fit, Feols)
    np.testing.assert_allclose(fit.coef().to_numpy(), fit_api.coef().to_numpy())
    np.testing.assert_allclose(fit.se().to_numpy(), fit_api.se().to_numpy())
