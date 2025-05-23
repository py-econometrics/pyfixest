import numpy as np
import pandas as pd
import pyhdfe
import pytest

from pyfixest.core import demean as demean_rs
from pyfixest.estimation.demean_ import _set_demeaner_backend, demean, demean_model
from pyfixest.estimation.jax.demean_jax_ import demean_jax


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
def test_demean(benchmark, demean_func):
    rng = np.random.default_rng(929291)

    N = 10_000
    M = 100
    x = rng.normal(0, 1, M * N).reshape((N, M))
    f1 = rng.choice(list(range(M)), N).reshape((N, 1))
    f2 = rng.choice(list(range(M)), N).reshape((N, 1))

    flist = np.concatenate((f1, f2), axis=1).astype(np.uint)

    # without weights
    weights = np.ones(N)
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, success = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe, res_pyfixest)

    # with weights
    weights = rng.uniform(0, 1, N).reshape((N, 1))
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weights)
    res_pyfixest, success = benchmark(
        demean_func, x, flist, weights.flatten(), tol=1e-10
    )
    assert np.allclose(res_pyhdfe, res_pyfixest)


def test_set_demeaner_backend():
    # Test numba backend
    demean_func = _set_demeaner_backend("numba")
    assert demean_func == demean

    # Test jax backend
    demean_func = _set_demeaner_backend("jax")
    assert demean_func == demean_jax

    # Test invalid backend raises ValueError
    with pytest.raises(ValueError, match="Invalid demeaner backend: invalid"):
        _set_demeaner_backend("invalid")


def test_demean_model_no_fixed_effects():
    """Test demean_model when there are no fixed effects."""
    # Create sample data
    N = 1000
    Y = pd.DataFrame({"y": np.random.randn(N)})
    X = pd.DataFrame({"x1": np.random.randn(N), "x2": np.random.randn(N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Test without fixed effects
    Yd, Xd = demean_model(
        Y=Y,
        X=X,
        fe=None,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # When no fixed effects, output should equal input
    assert np.allclose(Y.values, Yd.values)
    assert np.allclose(X.values, Xd.values)
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)


def test_demean_model_with_fixed_effects():
    """Test demean_model with fixed effects."""
    # Create sample data
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N), "fe2": rng.integers(0, 5, N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Run demean_model
    Yd, Xd = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Verify results are different from input (since we're demeaning)
    assert not np.allclose(Y.values, Yd.values)
    assert not np.allclose(X.values, Xd.values)

    # Verify column names are preserved
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)

    # Verify results are cached in lookup_dict
    assert "test" in lookup_dict
    cached_data = lookup_dict["test"][1]
    assert np.allclose(cached_data[Y.columns].values, Yd.values)
    assert np.allclose(cached_data[X.columns].values, Xd.values)


def test_demean_model_with_weights():
    """Test demean_model with weights."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = rng.uniform(0.5, 1.5, N)
    lookup_dict = {}

    # Run with weights
    Yd, Xd = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Run without weights for comparison
    Yd_unweighted, Xd_unweighted = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=np.ones(N),
        lookup_demeaned_data={},
        na_index_str="test2",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Results should be different with weights vs without
    assert not np.allclose(Yd.values, Yd_unweighted.values)
    assert not np.allclose(Xd.values, Xd_unweighted.values)


def test_demean_model_caching():
    """Test the caching behavior of demean_model."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = np.ones(N)
    lookup_dict = {}

    # First run - should compute and cache
    Yd1, Xd1 = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Second run - should use cache
    Yd2, Xd2 = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Results should be identical
    assert np.allclose(Yd1.values, Yd2.values)
    assert np.allclose(Xd1.values, Xd2.values)

    # Add new variable and verify partial caching
    X_new = X.copy()
    X_new["x3"] = rng.normal(0, 1, N)

    Yd3, Xd3 = demean_model(
        Y=Y,
        X=X_new,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index_str="test",
        fixef_tol=1e-8,
        demeaner_backend="numba",
    )

    # Original columns should match previous results
    assert np.allclose(Xd3[["x1", "x2"]].values, Xd2.values)
    # New column should be different
    assert "x3" in Xd3.columns
