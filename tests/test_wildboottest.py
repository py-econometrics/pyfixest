"""
Tests for Wild Cluster Bootstrap inference (Cameron, Gelbach & Miller 2008).

pyfixest/inference/wildboottest.py exposes two public functions:

  wildboottest(model, param, ...)          -- pyfixest Feols model wrapper
  wildboottest_numpy(Y, X, cluster_ids, param_idx, ...)  -- pure numpy API

Import note
-----------
The repo contains a pyfixest/ tree that requires a compiled Rust extension
(_core_impl) which cannot be built in this environment.  Pytest inserts the
repo root into sys.path, which would shadow the installed wheel and cause
ModuleNotFoundError.

To work around this we:
  1. Load wildboottest.py directly via importlib (no package machinery).
  2. Replace pyfixest-model tests with a lightweight stub that mimics the
     attributes wildboottest() reads (_coefnames, _clustervar, _cluster_df,
     _X, _Y), so no actual pyfixest import is needed for those tests.

The installed pyfixest wheel is NOT imported anywhere in this test file.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Load the module under test directly from its source file.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_WCB_PATH = os.path.join(_REPO_ROOT, "pyfixest", "inference", "wildboottest.py")

_spec = importlib.util.spec_from_file_location("_wcb", _WCB_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

wildboottest = _mod.wildboottest
wildboottest_numpy = _mod.wildboottest_numpy


# ---------------------------------------------------------------------------
# Stub model — mimics the pyfixest Feols attributes wildboottest() reads.
# ---------------------------------------------------------------------------

class _StubModel:
    """Minimal stand-in for a fitted pyfixest Feols model."""

    def __init__(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        cluster_ids: np.ndarray,
        coefnames: list[str],
        clustervar: str = "cluster",
    ):
        self._Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        self._X = X
        self._coefnames = coefnames
        self._clustervar = [clustervar]
        self._cluster_df = pd.DataFrame({clustervar: cluster_ids})

    @classmethod
    def from_dgp(
        cls,
        N: int = 200,
        G: int = 20,
        true_beta: float = 2.0,
        seed: int = 12345,
    ) -> "_StubModel":
        rng = np.random.default_rng(seed)
        cluster = np.repeat(np.arange(G), N // G)
        x = rng.standard_normal(N)
        y = true_beta * x + rng.standard_normal(N)
        X = np.column_stack([np.ones(N), x])
        return cls(Y=y, X=X, cluster_ids=cluster,
                   coefnames=["Intercept", "x"])


def _stub() -> _StubModel:
    """Significant model: true beta_x = 2."""
    return _StubModel.from_dgp(true_beta=2.0, seed=12345)


def _null_stub() -> _StubModel:
    """Null model: true beta_x = 0."""
    return _StubModel.from_dgp(true_beta=0.0, seed=99999)


def _no_cluster_stub() -> _StubModel:
    """Stub where _clustervar is empty (simulates no-cluster model)."""
    m = _StubModel.from_dgp(true_beta=1.0, seed=0)
    m._clustervar = []       # wildboottest should reject this
    m._cluster_df = None
    return m


# ---------------------------------------------------------------------------
# 1. Return structure
# ---------------------------------------------------------------------------

def test_wildboottest_returns_dict_with_expected_keys():
    result = wildboottest(_stub(), param="x", B=99, seed=0)
    assert isinstance(result, dict)
    assert {"param", "t_stat", "p_value", "B", "weights_type"} == set(result.keys())


def test_wildboottest_param_name_in_result():
    result = wildboottest(_stub(), param="x", B=99, seed=0)
    assert result["param"] == "x"


def test_wildboottest_B_in_result():
    result = wildboottest(_stub(), param="x", B=99, seed=0)
    assert result["B"] == 99


def test_wildboottest_weights_type_in_result():
    result = wildboottest(_stub(), param="x", B=99,
                          weights_type="rademacher", seed=0)
    assert result["weights_type"] == "rademacher"


# ---------------------------------------------------------------------------
# 2. p-value range
# ---------------------------------------------------------------------------

def test_wildboottest_pvalue_in_range():
    result = wildboottest(_stub(), param="x", B=199, seed=1)
    assert 0.0 <= result["p_value"] <= 1.0


def test_wildboottest_mammen_pvalue_in_range():
    result = wildboottest(_stub(), param="x", B=199,
                          weights_type="mammen", seed=1)
    assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Statistical power / size
# ---------------------------------------------------------------------------

def test_wildboottest_null_is_false():
    """Large true coefficient (beta=2) -> bootstrap p-value should be small."""
    result = wildboottest(_stub(), param="x", B=499, seed=42)
    assert result["p_value"] < 0.05, (
        f"Expected p < 0.05 for true coef=2; got {result['p_value']}"
    )


def test_wildboottest_null_is_true():
    """Zero true coefficient -> p-value should NOT be systematically tiny."""
    result = wildboottest(_null_stub(), param="x", B=999, seed=0)
    assert result["p_value"] > 0.01, (
        f"p-value suspiciously small when null is true: {result['p_value']}"
    )


# ---------------------------------------------------------------------------
# 4. Weight distribution variants
# ---------------------------------------------------------------------------

def test_wildboottest_rademacher_weights():
    result = wildboottest(_stub(), param="x", B=99,
                          weights_type="rademacher", seed=7)
    assert isinstance(result["p_value"], float)


def test_wildboottest_mammen_weights():
    result = wildboottest(_stub(), param="x", B=99,
                          weights_type="mammen", seed=7)
    assert isinstance(result["p_value"], float)


def test_wildboottest_invalid_weights_type_raises():
    with pytest.raises(ValueError, match="weights_type"):
        wildboottest(_stub(), param="x", B=99,
                     weights_type="gaussian", seed=0)


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

def test_wildboottest_reproducible():
    r1 = wildboottest(_stub(), param="x", B=199, seed=123)
    r2 = wildboottest(_stub(), param="x", B=199, seed=123)
    assert r1["p_value"] == r2["p_value"]
    assert r1["t_stat"] == r2["t_stat"]


def test_wildboottest_different_seeds_both_valid():
    r1 = wildboottest(_stub(), param="x", B=199, seed=1)
    r2 = wildboottest(_stub(), param="x", B=199, seed=9999)
    assert 0.0 <= r1["p_value"] <= 1.0
    assert 0.0 <= r2["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# 6. B iterations (pure-numpy API)
# ---------------------------------------------------------------------------

def test_wildboottest_B_iterations():
    rng = np.random.default_rng(0)
    N, G = 100, 10
    cluster = np.repeat(np.arange(G), N // G)
    x = rng.standard_normal(N)
    y = x + rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    for B in [49, 99, 199]:
        result = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                                    param_idx=1, B=B, seed=0)
        assert result["B"] == B


# ---------------------------------------------------------------------------
# 7. Pure-numpy API
# ---------------------------------------------------------------------------

def test_wildboottest_numpy_keys():
    rng = np.random.default_rng(0)
    N, G = 100, 10
    cluster = np.repeat(np.arange(G), N // G)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    y = 2.0 * X[:, 1] + rng.standard_normal(N)
    result = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                                param_idx=1, B=99, seed=0)
    assert set(result.keys()) == {"param_idx", "t_stat", "p_value", "B", "weights_type"}


def test_wildboottest_numpy_pvalue_range():
    rng = np.random.default_rng(42)
    N, G = 100, 10
    cluster = np.repeat(np.arange(G), N // G)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    y = rng.standard_normal(N)
    result = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                                param_idx=1, B=199, seed=42)
    assert 0.0 <= result["p_value"] <= 1.0


def test_wildboottest_numpy_significant():
    """Strong signal -> small p-value via numpy API."""
    rng = np.random.default_rng(0)
    N, G = 200, 20
    cluster = np.repeat(np.arange(G), N // G)
    x = rng.standard_normal(N)
    y = 5.0 * x + 0.1 * rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    result = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                                param_idx=1, B=499, seed=0)
    assert result["p_value"] < 0.05


def test_wildboottest_numpy_out_of_bounds_param_idx():
    rng = np.random.default_rng(0)
    N = 50
    cluster = np.repeat(np.arange(5), N // 5)
    X = np.column_stack([np.ones(N), rng.standard_normal(N)])
    y = rng.standard_normal(N)
    with pytest.raises(IndexError):
        wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                           param_idx=5, B=49, seed=0)


def test_wildboottest_numpy_mammen():
    rng = np.random.default_rng(7)
    N, G = 100, 10
    cluster = np.repeat(np.arange(G), N // G)
    x = rng.standard_normal(N)
    y = 2.0 * x + rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    result = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                                param_idx=1, B=99,
                                weights_type="mammen", seed=7)
    assert 0.0 <= result["p_value"] <= 1.0


def test_wildboottest_numpy_reproducible():
    rng = np.random.default_rng(0)
    N, G = 100, 10
    cluster = np.repeat(np.arange(G), N // G)
    x = rng.standard_normal(N)
    y = x + rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x])
    r1 = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                            param_idx=1, B=99, seed=55)
    r2 = wildboottest_numpy(Y=y, X=X, cluster_ids=cluster,
                            param_idx=1, B=99, seed=55)
    assert r1["p_value"] == r2["p_value"]


# ---------------------------------------------------------------------------
# 8. Model wrapper: error paths (use stubs, no real pyfixest needed)
# ---------------------------------------------------------------------------

def test_wildboottest_bad_param_raises():
    with pytest.raises(ValueError, match="not found"):
        wildboottest(_stub(), param="z_nonexistent", B=49, seed=0)


def test_wildboottest_no_cluster_raises():
    """Model without cluster info should raise ValueError."""
    with pytest.raises(ValueError, match="cluster"):
        wildboottest(_no_cluster_stub(), param="x", B=49, seed=0)
