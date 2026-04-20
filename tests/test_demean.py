import numpy as np
import pandas as pd
import pyhdfe
import pytest

import pyfixest as pf
from pyfixest.core import demean as demean_rs
from pyfixest.core.demean import demean_within
from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32, demean_cupy64
from pyfixest.estimation.internals.backends import BACKENDS
from pyfixest.estimation.internals.demean_ import (
    _set_demeaner_backend,
    demean,
    demean_model,
)
from pyfixest.estimation.jax.demean_jax_ import demean_jax
from tests._torch_test_utils import HAS_TORCH, torch_param

GENERIC_DEMEAN_FUNCS = [
    pytest.param(demean, id="demean_numba"),
    pytest.param(demean_jax, id="demean_jax"),
    pytest.param(demean_rs, id="demean_rs"),
    pytest.param(demean_cupy32, id="demean_cupy32"),
    pytest.param(demean_cupy64, id="demean_cupy64"),
]

if HAS_TORCH:
    from pyfixest.estimation.torch.demean_torch_ import demean_torch

    GENERIC_DEMEAN_FUNCS.append(pytest.param(demean_torch, id="demean_torch"))


TORCH_DEVICE_DEMEANERS = [
    torch_param(("demean_torch_cpu", 1e-6, 1e-8), id="demean_torch_cpu"),
    torch_param(("demean_torch_mps", 1e-3, 1e-3), id="demean_torch_mps", require="mps"),
    torch_param(
        ("demean_torch_cuda", 1e-6, 1e-8), id="demean_torch_cuda", require="cuda"
    ),
    torch_param(
        ("demean_torch_cuda32", 1e-3, 1e-3),
        id="demean_torch_cuda32",
        require="cuda",
    ),
]


@pytest.fixture(scope="module")
def demean_data():
    rng = np.random.default_rng(929291)

    n_obs = 1_000
    n_cols = 10
    x = rng.normal(0, 1, n_cols * n_obs).reshape((n_obs, n_cols))
    f1 = rng.choice(list(range(n_cols)), n_obs).reshape((n_obs, 1))
    f2 = rng.choice(list(range(n_cols)), n_obs).reshape((n_obs, 1))
    flist = np.concatenate((f1, f2), axis=1).astype(np.uint64)
    weights = rng.uniform(0, 1, n_obs)

    return x, flist, weights


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean(benchmark, demean_func, demean_data):
    x, flist, weighted = demean_data

    # without weights
    weights = np.ones(x.shape[0])
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, _ = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)

    # with weights
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weighted.reshape((x.shape[0], 1)))
    res_pyfixest, _ = benchmark(demean_func, x, flist, weighted, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)


@pytest.mark.parametrize(("backend_name", "rtol", "atol"), TORCH_DEVICE_DEMEANERS)
def test_torch_device_backends_match_pyhdfe(backend_name, rtol, atol, demean_data):
    from pyfixest.estimation.torch.demean_torch_ import (
        demean_torch_cpu,
        demean_torch_cuda,
        demean_torch_cuda32,
        demean_torch_mps,
    )

    backend_map = {
        "demean_torch_cpu": demean_torch_cpu,
        "demean_torch_mps": demean_torch_mps,
        "demean_torch_cuda": demean_torch_cuda,
        "demean_torch_cuda32": demean_torch_cuda32,
    }

    x, flist, weights = demean_data
    demean_func = backend_map[backend_name]

    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_torch, success = demean_func(x, flist, np.ones(x.shape[0]), tol=1e-10)
    assert success, f"{backend_name} did not converge on unweighted demeaning"
    np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=rtol, atol=atol)

    res_pyhdfe = algorithm.residualize(x, weights.reshape((x.shape[0], 1)))
    res_torch, success = demean_func(x, flist, weights, tol=1e-10)
    assert success, f"{backend_name} did not converge on weighted demeaning"
    np.testing.assert_allclose(res_torch, res_pyhdfe, rtol=rtol, atol=atol)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_sparse_dummy_reencodes_non_contiguous_groups():
    from pyfixest.estimation.torch._sparse_dummy import _build_sparse_dummy
    from tests._torch_test_utils import torch

    flist = np.array([[2], [9], [2], [5]], dtype=np.uint64)

    D = _build_sparse_dummy(flist, torch.device("cpu"), torch.float64).to_dense()
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    assert torch.equal(D, expected)


def test_set_demeaner_backend():
    for backend in [
        "numba",
        "jax",
        "rust",
        "cupy32",
        "cupy64",
        "scipy",
        "torch",
        "torch_cpu",
        "torch_mps",
        "torch_cuda",
        "torch_cuda32",
    ]:
        if backend.startswith("torch"):
            with pytest.warns(UserWarning, match="experimental torch algorithms"):
                demean_func = _set_demeaner_backend(backend)
        else:
            demean_func = _set_demeaner_backend(backend)
        assert demean_func == BACKENDS[backend]["demean"]

    demean_func = _set_demeaner_backend("rust-cg")
    assert demean_func == demean_within

    # Test invalid backend raises ValueError
    with pytest.raises(ValueError, match="Invalid demeaner backend: invalid"):
        _set_demeaner_backend("invalid")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_feols_warns_for_experimental_torch_backend():
    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 0.0, 1.0],
            "fe": [0, 0, 1, 1],
        }
    )

    with pytest.warns(UserWarning, match="experimental torch algorithms"):
        pf.feols("y ~ x | fe", data=data, demeaner_backend="torch_cpu")


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_no_fixed_effects(benchmark, demean_func):
    """Test demean_model when there are no fixed effects."""
    # Create sample data
    N = 1000
    Y = pd.DataFrame({"y": np.random.randn(N)})
    X = pd.DataFrame({"x1": np.random.randn(N), "x2": np.random.randn(N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Test without fixed effects
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=None,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # When no fixed effects, output should equal input
    assert np.allclose(Y.values, Yd.values)
    assert np.allclose(X.values, Xd.values)
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_with_fixed_effects(benchmark, demean_func):
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
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Verify results are different from input (since we're demeaning)
    assert not np.allclose(Y.values, Yd.values)
    assert not np.allclose(X.values, Xd.values)

    # Verify column names are preserved
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)

    # Verify results are cached in lookup_dict
    assert frozenset() in lookup_dict
    cached_data = lookup_dict[frozenset()][1]
    assert np.allclose(cached_data[Y.columns].values, Yd.values)
    assert np.allclose(cached_data[X.columns].values, Xd.values)


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_with_weights(benchmark, demean_func):
    """Test demean_model with weights."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = rng.uniform(0.5, 1.5, N)
    lookup_dict = {}

    # Run with weights
    Yd, Xd = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Run without weights for comparison (fresh lookup dict to avoid cache hit)
    Yd_unweighted, Xd_unweighted = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=np.ones(N),
        lookup_demeaned_data={},
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Results should be different with weights vs without
    assert not np.allclose(Yd.values, Yd_unweighted.values)
    assert not np.allclose(Xd.values, Xd_unweighted.values)


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_caching(benchmark, demean_func):
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
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Second run - should use cache
    Yd2, Xd2 = benchmark(
        demean_model,
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Results should be identical
    assert np.allclose(Yd1.values, Yd2.values)
    assert np.allclose(Xd1.values, Xd2.values)

    # Add new variable and verify partial caching
    X_new = X.copy()
    X_new["x3"] = rng.normal(0, 1, N)

    _, Xd3 = demean_model(
        Y=Y,
        X=X_new,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Original columns should match previous results
    assert np.allclose(Xd3[["x1", "x2"]].values, Xd2.values)
    # New column should be different
    assert "x3" in Xd3.columns


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_maxiter_convergence_failure(demean_func):
    """Test that demean_model fails when maxiter is too small."""
    N = 100
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N)})
    # Many fixed effects to make convergence difficult
    fe = pd.DataFrame(
        {"fe1": rng.choice(N // 10, N), "fe2": rng.choice(N // 10, N)}
    )  # Each obs is its own FE
    weights = np.ones(N)
    lookup_dict = {}

    # Should fail with very small maxiter
    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        demean_model(
            Y=Y,
            X=X,
            fe=fe,
            weights=weights,
            lookup_demeaned_data=lookup_dict,
            na_index=frozenset(),
            fixef_tol=1e-6,
            fixef_maxiter=1,  # Very small limit
            demean_func=demean_func,
        )


@pytest.mark.parametrize("demean_func", GENERIC_DEMEAN_FUNCS)
def test_demean_model_custom_maxiter_success(demean_func):
    """Test that demean_model succeeds with reasonable maxiter."""
    N = 1000
    rng = np.random.default_rng(42)

    Y = pd.DataFrame({"y": rng.normal(0, 1, N)})
    X = pd.DataFrame({"x1": rng.normal(0, 1, N)})
    fe = pd.DataFrame({"fe1": rng.integers(0, 10, N)})
    weights = np.ones(N)
    lookup_dict = {}

    # Should succeed with reasonable maxiter
    Yd, Xd = demean_model(
        Y=Y,
        X=X,
        fe=fe,
        weights=weights,
        lookup_demeaned_data=lookup_dict,
        na_index=frozenset(),
        fixef_tol=1e-6,
        fixef_maxiter=5000,  # Custom limit
        demean_func=demean_func,
    )

    # Just verify it returns valid results
    assert isinstance(Yd, pd.DataFrame)
    assert isinstance(Xd, pd.DataFrame)
    assert Yd.shape == Y.shape
    assert Xd.shape == X.shape


def test_demean_maxiter_parameter():
    """Test that the demean function respects maxiter parameter."""
    N = 100
    rng = np.random.default_rng(42)

    # Create data that's hard to converge
    x = rng.normal(0, 1, N * 2).reshape((N, 2))
    flist = np.arange(N).reshape((N, 1)).astype(np.uint)  # Many FEs
    weights = np.ones(N)

    # Test with very small maxiter
    _, success = demean(x, flist, weights, tol=1e-10, maxiter=1)
    assert not success  # Should fail to converge

    # Test with large maxiter
    _, success = demean(x, flist, weights, tol=1e-10, maxiter=100_000)
    # May or may not converge, but shouldn't crash


def test_feols_integration_maxiter():
    """Integration test: Test fixef_maxiter flows from feols to demean."""
    import pyfixest as pf

    N = 1000  # More observations
    rng = np.random.default_rng(42)

    # Create data with many (but not N) fixed effects
    data = pd.DataFrame(
        {
            "y": rng.normal(0, 1, N),
            "x": rng.normal(0, 1, N),
            "fe": rng.integers(0, 50, N),  # 50 fixed effects, not N
        }
    )

    # Should fail with tiny maxiter
    with pytest.raises(ValueError, match="Demeaning failed after 1 iterations"):
        pf.feols("y ~ x | fe", data=data, fixef_maxiter=1)

    # Should work with default
    model = pf.feols("y ~ x | fe", data=data)
    assert model is not None


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean_rs, demean_within, demean_cupy32, demean_cupy64],
    ids=["demean_rs", "demean_within", "demean_cupy32", "demean_cupy64"],
)
def test_demean_complex_fixed_effects(benchmark, demean_func):
    """Benchmark demean functions with complex multi-level fixed effects."""
    X, flist, weights = generate_complex_fixed_effects_data()

    X_demeaned, success = benchmark.pedantic(
        demean_func,
        args=(X, flist, weights),
        kwargs={"tol": 1e-10},
        iterations=1,
        rounds=1,
        warmup_rounds=0,
    )

    assert success, "Benchmarked demeaning should succeed"
    assert X_demeaned.shape == X.shape


def generate_complex_fixed_effects_data():
    """
    Complex fixed effects example ported from fixest R-implementation:
    https://github.com/lrberge/fixest/blob/ac1be27fda5fc381c0128b861eaf5bda88af846c/_BENCHMARK/Data%20generation.R#L125 .

    """
    rng = np.random.default_rng(42)
    n = 10**5  # Large dataset for benchmarking
    nb_indiv = n // 20
    nb_firm = max(1, round(n / 160))
    nb_year = max(1, round(n**0.3))
    # Generate fixed effect IDs
    id_indiv = rng.choice(nb_indiv, n, replace=True)
    id_firm_base = rng.integers(0, 21, n) + np.maximum(1, id_indiv // 8 - 10)
    id_firm = np.minimum(id_firm_base, nb_firm - 1)
    id_year = rng.choice(nb_year, n, replace=True)
    # Create variables
    x1 = (
        5 * np.cos(id_indiv)
        + 5 * np.sin(id_firm)
        + 5 * np.sin(id_year)
        + rng.uniform(0, 1, n)
    )
    x2 = np.cos(id_indiv) + np.sin(id_firm) + np.sin(id_year) + rng.normal(0, 1, n)
    y = (
        3 * x1
        + 5 * x2
        + np.cos(id_indiv)
        + np.cos(id_firm) ** 2
        + np.sin(id_year)
        + rng.normal(0, 1, n)
    )
    X = np.column_stack([x1, x2, y])
    flist = np.column_stack([id_indiv, id_firm, id_year]).astype(np.uint64)
    weights = rng.uniform(0.5, 2.0, n)
    return X, flist, weights
