import numpy as np
import pandas as pd
import pyhdfe
import pytest

from pyfixest.core._core_impl import _demean_rs
from pyfixest.core.demean import demean as demean_rs
from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32, demean_cupy64
from pyfixest.estimation.demean_ import _set_demeaner_backend, demean, demean_model
from pyfixest.estimation.jax.demean_jax_ import demean_jax


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
)
def test_demean(benchmark, demean_func):
    rng = np.random.default_rng(929291)

    N = 1_000
    M = 10
    x = rng.normal(0, 1, M * N).reshape((N, M))
    f1 = rng.choice(list(range(M)), N).reshape((N, 1))
    f2 = rng.choice(list(range(M)), N).reshape((N, 1))

    flist = np.concatenate((f1, f2), axis=1).astype(np.uint)

    # without weights
    weights = np.ones(N)
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x)
    res_pyfixest, _ = demean_func(x, flist, weights, tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)

    # with weights
    weights = rng.uniform(0, 1, N).reshape((N, 1))
    algorithm = pyhdfe.create(flist)
    res_pyhdfe = algorithm.residualize(x, weights)
    res_pyfixest, _ = benchmark(demean_func, x, flist, weights.flatten(), tol=1e-10)
    assert np.allclose(res_pyhdfe[10, 0:], res_pyfixest[10, 0:], rtol=1e-06, atol=1e-08)


def test_set_demeaner_backend():
    # Test numba backend
    demean_func = _set_demeaner_backend("numba")
    assert demean_func == demean

    # Test jax backend
    demean_func = _set_demeaner_backend("jax")
    assert demean_func == demean_jax

    demean_func = _set_demeaner_backend("rust")
    assert demean_func == demean_rs

    demean_func = _set_demeaner_backend("cupy32")
    assert demean_func == demean_cupy32

    demean_func = _set_demeaner_backend("cupy64")
    assert demean_func == demean_cupy64

    # Test invalid backend raises ValueError
    with pytest.raises(ValueError, match="Invalid demeaner backend: invalid"):
        _set_demeaner_backend("invalid")


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # When no fixed effects, output should equal input
    assert np.allclose(Y.values, Yd.values)
    assert np.allclose(X.values, Xd.values)
    assert Yd.columns.equals(Y.columns)
    assert Xd.columns.equals(X.columns)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
    assert "test" in lookup_dict
    cached_data = lookup_dict["test"][1]
    assert np.allclose(cached_data[Y.columns].values, Yd.values)
    assert np.allclose(cached_data[X.columns].values, Xd.values)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
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
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Results should be different with weights vs without
    assert not np.allclose(Yd.values, Yd_unweighted.values)
    assert not np.allclose(Xd.values, Xd_unweighted.values)


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
        na_index_str="test",
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
        na_index_str="test",
        fixef_tol=1e-6,
        fixef_maxiter=10_000,
        demean_func=demean_func,
    )

    # Original columns should match previous results
    assert np.allclose(Xd3[["x1", "x2"]].values, Xd2.values)
    # New column should be different
    assert "x3" in Xd3.columns


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
            fixef_tol=1e-6,
            fixef_maxiter=1,  # Very small limit
            demean_func=demean_func,
        )


@pytest.mark.parametrize(
    argnames="demean_func",
    argvalues=[
        demean,
        demean_jax,
        demean_rs,
        demean_cupy32,
        demean_cupy64,
    ],
    ids=[
        "demean_numba",
        "demean_jax",
        "demean_rs",
        "demean_cupy32",
        "demean_cupy64",
    ],
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
    argvalues=[demean_rs, demean_cupy32, demean_cupy64],
    ids=["demean_rs", "demean_cupy32", "demean_cupy64"],
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


# =============================================================================
# FE Coefficient Tests
# =============================================================================


def _apply_fe_coefficients(x, flist, fe_coefficients, n_groups):
    """
    Apply FE coefficients to reconstruct residuals.

    Returns x[i] - sum_q(coef[fe_q[i]]) for each observation.
    """
    n_obs, n_features = x.shape
    n_fe = flist.shape[1]

    # Compute coefficient offsets for each FE
    coef_offsets = np.zeros(n_fe, dtype=int)
    for q in range(1, n_fe):
        coef_offsets[q] = coef_offsets[q - 1] + n_groups[q - 1]

    reconstructed = np.zeros_like(x)
    for k in range(n_features):
        for i in range(n_obs):
            fe_sum = 0.0
            for q in range(n_fe):
                g = int(flist[i, q])
                fe_sum += fe_coefficients[coef_offsets[q] + g, k]
            reconstructed[i, k] = x[i, k] - fe_sum

    return reconstructed


def test_fe_coefficients_single_fe():
    """Test FE coefficients are correct for single FE."""
    n_obs = 100
    n_groups = 10

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_obs, 2))
    flist = (np.arange(n_obs) % n_groups).reshape(-1, 1).astype(np.uint64)
    weights = np.ones(n_obs)

    result = _demean_rs(x, flist, weights)

    assert result["success"], "Should converge"
    assert result["fe_coefficients"].shape == (n_groups, 2), "Wrong coefficient shape"

    # Verify coefficients: applying them should give same residuals as demeaned
    reconstructed = _apply_fe_coefficients(
        x, flist, result["fe_coefficients"], [n_groups]
    )

    np.testing.assert_allclose(
        result["demeaned"],
        reconstructed,
        rtol=1e-10,
        atol=1e-10,
        err_msg="FE coefficients don't reconstruct demeaned values",
    )


def test_fe_coefficients_two_fe():
    """Test FE coefficients are correct for two FEs."""
    n_obs = 100
    n_groups_0 = 10
    n_groups_1 = 5

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_obs, 3))
    flist = np.column_stack(
        [np.arange(n_obs) % n_groups_0, np.arange(n_obs) % n_groups_1]
    ).astype(np.uint64)
    weights = np.ones(n_obs)

    result = _demean_rs(x, flist, weights)

    assert result["success"], "Should converge"
    assert result["fe_coefficients"].shape == (n_groups_0 + n_groups_1, 3)

    # Verify coefficients reconstruct demeaned values
    reconstructed = _apply_fe_coefficients(
        x, flist, result["fe_coefficients"], [n_groups_0, n_groups_1]
    )

    np.testing.assert_allclose(
        result["demeaned"],
        reconstructed,
        rtol=1e-8,
        atol=1e-8,
        err_msg="FE coefficients don't reconstruct demeaned values",
    )


def test_fe_coefficients_ordering():
    """Test that FE coefficients are in original FE order, not reordered."""
    n_obs = 100

    # FE 0: 5 groups (smaller), FE 1: 20 groups (larger)
    # Internally, FEs get reordered by size (largest first)
    # But coefficients should be returned in original order
    n_groups_0 = 5  # smaller
    n_groups_1 = 20  # larger

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_obs, 2))
    flist = np.column_stack(
        [np.arange(n_obs) % n_groups_0, np.arange(n_obs) % n_groups_1]
    ).astype(np.uint64)
    weights = np.ones(n_obs)

    result = _demean_rs(x, flist, weights)

    # Verify coefficient shape matches original order
    assert result["fe_coefficients"].shape == (n_groups_0 + n_groups_1, 2)

    # Verify coefficients work with original flist ordering
    reconstructed = _apply_fe_coefficients(
        x, flist, result["fe_coefficients"], [n_groups_0, n_groups_1]
    )

    np.testing.assert_allclose(
        result["demeaned"],
        reconstructed,
        rtol=1e-8,
        atol=1e-8,
        err_msg="Coefficients may be in wrong order",
    )


def test_fe_coefficients_three_fe():
    """Test FE coefficients are correct for three FEs."""
    n_obs = 120

    # Create FEs with different sizes to trigger reordering
    # Original: FE0=3 groups (smallest), FE1=15 groups (largest), FE2=8 groups (middle)
    n_groups_0 = 3
    n_groups_1 = 15
    n_groups_2 = 8

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_obs, 2))
    flist = np.column_stack(
        [
            np.arange(n_obs) % n_groups_0,
            np.arange(n_obs) % n_groups_1,
            np.arange(n_obs) % n_groups_2,
        ]
    ).astype(np.uint64)
    weights = np.ones(n_obs)

    result = _demean_rs(x, flist, weights)

    assert result["success"], "Should converge"
    assert result["fe_coefficients"].shape == (
        n_groups_0 + n_groups_1 + n_groups_2,
        2,
    )

    # Verify coefficients reconstruct demeaned values
    reconstructed = _apply_fe_coefficients(
        x, flist, result["fe_coefficients"], [n_groups_0, n_groups_1, n_groups_2]
    )

    np.testing.assert_allclose(
        result["demeaned"],
        reconstructed,
        rtol=1e-6,
        atol=1e-6,
        err_msg="FE coefficients don't reconstruct demeaned values",
    )


def test_fe_coefficients_weighted():
    """Test FE coefficients are correct with non-uniform weights."""
    n_obs = 100
    n_groups_0 = 10
    n_groups_1 = 5

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (n_obs, 2))
    flist = np.column_stack(
        [np.arange(n_obs) % n_groups_0, np.arange(n_obs) % n_groups_1]
    ).astype(np.uint64)
    weights = rng.uniform(0.5, 2.0, n_obs)

    result = _demean_rs(x, flist, weights)

    assert result["success"], "Should converge"

    # Verify coefficients reconstruct demeaned values
    reconstructed = _apply_fe_coefficients(
        x, flist, result["fe_coefficients"], [n_groups_0, n_groups_1]
    )

    np.testing.assert_allclose(
        result["demeaned"],
        reconstructed,
        rtol=1e-8,
        atol=1e-8,
        err_msg="Weighted FE coefficients don't reconstruct demeaned values",
    )
