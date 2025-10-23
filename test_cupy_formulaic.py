"""
Test CuPy backend with formulaic sparse matrix creation.

This demonstrates the improved accuracy and efficiency of using formulaic
to create sparse FE matrices with proper reference level handling.
"""

import numpy as np
import pandas as pd

from pyfixest.core import demean as demean_rs
from pyfixest.estimation.cupy.demean_cupy_ import (
    create_fe_sparse_matrix,
    demean_cupy,
)


def test_create_fe_sparse_matrix():
    """Test sparse FE matrix creation with formulaic."""
    # Simple case: 2 FEs
    fe = pd.DataFrame({"fe1": [0, 0, 1, 1], "fe2": [0, 1, 0, 1]})

    # With reference level dropped (default)
    D = create_fe_sparse_matrix(fe, drop_reference=True)
    print(f"\nWith drop_reference=True:")
    print(f"Shape: {D.shape}")  # Should be (4, 3) - 2 levels for fe1 (drop 1), 2 for fe2 (drop 1)
    print(f"Sparse format: {D.format}")
    print(f"Density: {D.nnz / (D.shape[0] * D.shape[1]):.2%}")
    print(f"Matrix:\n{D.toarray()}")

    # Without reference level dropped
    D_full = create_fe_sparse_matrix(fe, drop_reference=False)
    print(f"\nWith drop_reference=False:")
    print(f"Shape: {D_full.shape}")  # Should be (4, 4) - 2 levels for fe1, 2 for fe2
    print(f"Matrix:\n{D_full.toarray()}")


def test_demean_cupy_with_fe_dataframe():
    """Test demean_cupy with fe DataFrame vs flist."""
    rng = np.random.default_rng(42)
    N = 1000

    # Create data
    x = rng.normal(0, 1, N * 2).reshape((N, 2))
    fe1 = rng.integers(0, 10, N)
    fe2 = rng.integers(0, 5, N)
    weights = rng.uniform(0.5, 1.5, N)

    # Old way: using flist
    flist = np.column_stack([fe1, fe2]).astype(np.uint64)
    x_demeaned_old, success_old = demean_cupy(x, flist=flist, weights=weights)
    print(f"\nOld way (flist): success={success_old}")

    # New way: using fe DataFrame
    fe = pd.DataFrame({"fe1": fe1, "fe2": fe2})
    x_demeaned_new, success_new = demean_cupy(
        x, fe=fe, weights=weights, drop_reference=True
    )
    print(f"New way (fe DataFrame): success={success_new}")

    # Compare with Rust backend
    x_demeaned_rust, success_rust = demean_rs(x, flist, weights, tol=1e-10)
    print(f"Rust backend: success={success_rust}")

    # Check agreement
    max_diff_new = np.abs(x_demeaned_new - x_demeaned_rust).max()
    max_diff_old = np.abs(x_demeaned_old - x_demeaned_rust).max()
    print(f"\nMax diff (new vs rust): {max_diff_new:.2e}")
    print(f"Max diff (old vs rust): {max_diff_old:.2e}")

    # New way should match rust much better
    assert success_new
    assert success_rust
    # Allow slightly larger tolerance since we're using different encoding
    assert max_diff_new < 1e-6, f"New method differs too much: {max_diff_new}"


def test_demean_cupy_complex_fe():
    """Test with complex multi-level fixed effects."""
    from tests.test_demean import generate_complex_fixed_effects_data

    X, flist, weights = generate_complex_fixed_effects_data()

    # Extract just first 2 columns for demeaning test
    x = X[:, :2]

    # Old way
    x_demeaned_old, success_old = demean_cupy(x, flist=flist, weights=weights)
    print(f"\nComplex FE - Old way: success={success_old}")

    # New way: convert flist to DataFrame
    fe = pd.DataFrame(
        {f"fe{i}": flist[:, i] for i in range(flist.shape[1])}
    )
    x_demeaned_new, success_new = demean_cupy(x, fe=fe, weights=weights)
    print(f"Complex FE - New way: success={success_new}")

    # Rust reference
    x_demeaned_rust, success_rust = demean_rs(x, flist, weights, tol=1e-10)
    print(f"Complex FE - Rust: success={success_rust}")

    max_diff_new = np.abs(x_demeaned_new - x_demeaned_rust).max()
    max_diff_old = np.abs(x_demeaned_old - x_demeaned_rust).max()
    rel_err_new = max_diff_new / (np.abs(x_demeaned_rust).max() + 1e-10)
    rel_err_old = max_diff_old / (np.abs(x_demeaned_rust).max() + 1e-10)

    print(f"\nMax diff (new vs rust): {max_diff_new:.2e} (rel: {rel_err_new:.2e})")
    print(f"Max diff (old vs rust): {max_diff_old:.2e} (rel: {rel_err_old:.2e})")

    assert success_new
    # New way should have better relative error
    print(f"\nRelative error improvement: {rel_err_old / rel_err_new:.2f}x")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing formulaic sparse matrix creation")
    print("=" * 80)
    test_create_fe_sparse_matrix()

    print("\n" + "=" * 80)
    print("Testing demean_cupy with fe DataFrame")
    print("=" * 80)
    test_demean_cupy_with_fe_dataframe()

    print("\n" + "=" * 80)
    print("Testing complex fixed effects")
    print("=" * 80)
    test_demean_cupy_complex_fe()

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
