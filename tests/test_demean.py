import numpy as np
import pandas as pd
import pyhdfe
import pytest

from pyfixest.core import demean as demean_rs
from pyfixest.core.demean_accelerated import demean_accelerated
from pyfixest.estimation.demean_ import _set_demeaner_backend, demean, demean_model
from pyfixest.estimation.jax.demean_jax_ import demean_jax
from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs, demean_accelerated, demean_cupy],
    ids=["demean_numba", "demean_jax", "demean_rs", "demean_accelerated", "demean_cupy"],
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
    res_pyfixest, _ = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe, res_pyfixest)

    # with weights
    weights = rng.uniform(0, 1, N).reshape((N, 1))
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weights)
    res_pyfixest, _ = benchmark(demean_func, x, flist, weights.flatten(), tol=1e-10)
    assert np.allclose(res_pyhdfe, res_pyfixest)

    # Additional test for cupy with formulaic encoding
    if demean_func == demean_cupy:
        try:
            import formulaic  # noqa: F401

            # Test with fe DataFrame using formulaic
            fe = pd.DataFrame({f"fe{i}": flist[:, i] for i in range(flist.shape[1])})
            res_formulaic, success_fe = demean_cupy(x, fe=fe, weights=weights.flatten())
            assert success_fe, "CuPy with fe DataFrame should succeed"

            # Compare with pyhdfe reference
            assert np.allclose(res_pyhdfe, res_formulaic), \
                "CuPy formulaic should match pyhdfe"

            # Compare formulaic vs flist approach
            max_diff = np.abs(res_formulaic - res_pyfixest).max()
            print(f"\n  CuPy formulaic vs flist: max diff = {max_diff:.2e}")

        except ImportError:
            pass  # Skip if formulaic not available


def test_set_demeaner_backend():
    # Test numba backend
    demean_func = _set_demeaner_backend("numba")
    assert demean_func == demean

    # Test jax backend
    demean_func = _set_demeaner_backend("jax")
    assert demean_func == demean_jax

    demean_func = _set_demeaner_backend("rust")
    assert demean_func == demean_rs

    # Test invalid backend raises ValueError
    with pytest.raises(ValueError, match="Invalid demeaner backend: invalid"):
        _set_demeaner_backend("invalid")


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
        demean_func=demean_func,
    )

    # When no fixed effects, output should equal input
    assert np.allclose(Y.values, Yd.values)
    assert np.allclose(X.values, Xd.values)
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
        demean_func=demean_func,
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


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
        demean_func=demean_func,
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
        fixef_maxiter=100_000,
        demean_func=demean_func,
    )

    # Results should be different with weights vs without
    assert not np.allclose(Yd.values, Yd_unweighted.values)
    assert not np.allclose(Xd.values, Xd_unweighted.values)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
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
        na_index_str="test",
        fixef_tol=1e-8,
        fixef_maxiter=100_000,
        demean_func=demean_func,
    )

    # Original columns should match previous results
    assert np.allclose(Xd3[["x1", "x2"]].values, Xd2.values)
    # New column should be different
    assert "x3" in Xd3.columns


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
            na_index_str="test",
            fixef_tol=1e-8,
            fixef_maxiter=1,  # Very small limit
            demean_func=demean_func,
        )


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[demean, demean_jax, demean_rs],
    ids=["demean_numba", "demean_jax", "demean_rs"],
)
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
        na_index_str="test",
        fixef_tol=1e-8,
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
    argvalues=[demean_rs, demean_accelerated, demean_cupy],
    ids=["demean_rs", "demean_accelerated", "demean_cupy"],
)
def test_demean_complex_fixed_effects(benchmark, demean_func):
    """Benchmark demean functions with complex multi-level fixed effects."""
    X, flist, weights = generate_complex_fixed_effects_data()

    # For CuPy, use formulaic encoding for better accuracy
    if demean_func == demean_cupy:
        try:
            import formulaic  # noqa: F401

            # Create fe DataFrame for formulaic encoding
            fe = pd.DataFrame({f"fe{i}": flist[:, i] for i in range(flist.shape[1])})

            # Benchmark formulaic approach - benchmark the actual call
            def cupy_with_formulaic():
                return demean_cupy(X, fe=fe, weights=weights)

            X_demeaned, success = benchmark(cupy_with_formulaic)

            assert success, "CuPy with formulaic should succeed"
            assert X_demeaned.shape == X.shape

            # Also test flist approach for comparison (not benchmarked)
            X_demeaned_flist, success_flist = demean_cupy(X, flist=flist, weights=weights)
            assert success_flist, "CuPy with flist should succeed"

            # Compare formulaic vs flist
            max_diff = np.abs(X_demeaned - X_demeaned_flist).max()
            rel_err = max_diff / (np.abs(X_demeaned_flist).max() + 1e-10)
            print(f"\n  CuPy: formulaic vs flist (max diff: {max_diff:.2e}, rel err: {rel_err:.2e})")

        except ImportError:
            # Fall back to flist if formulaic not available
            X_demeaned, success = benchmark(demean_func, X, flist, weights, tol=1e-10)
            assert success, "Benchmarked demeaning should succeed"
            assert X_demeaned.shape == X.shape
    else:
        # For other backends, use standard flist approach
        X_demeaned, success = benchmark(demean_func, X, flist, weights, tol=1e-10)
        assert success, "Benchmarked demeaning should succeed"
        assert X_demeaned.shape == X.shape


def generate_complex_fixed_effects_data():
    """
    Complex fixed effects example ported from fixest R-implementation:
    https://github.com/lrberge/fixest/blob/ac1be27fda5fc381c0128b861eaf5bda88af846c/_BENCHMARK/Data%20generation.R#L125 .

    """
    rng = np.random.default_rng(42)
    n = 20**5  # Large dataset for benchmarking
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
