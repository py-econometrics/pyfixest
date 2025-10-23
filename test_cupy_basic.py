"""Basic test to verify CuPy demeaner implementation."""
import numpy as np
import pandas as pd

print("Testing CuPy demeaner implementation...")

# Test 1: Import
print("\n1. Testing imports...")
try:
    from pyfixest.estimation.cupy.demean_cupy_ import (
        CUPY_AVAILABLE,
        FORMULAIC_AVAILABLE,
        CupyFWLDemeaner,
        create_fe_sparse_matrix,
        demean_cupy,
    )
    print(f"   ✓ Imports successful")
    print(f"   CuPy available: {CUPY_AVAILABLE}")
    print(f"   Formulaic available: {FORMULAIC_AVAILABLE}")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create demeaner instance
print("\n2. Testing CupyFWLDemeaner instantiation...")
try:
    demeaner = CupyFWLDemeaner(use_gpu=False)  # Force CPU for testing
    print(f"   ✓ Demeaner created (use_gpu={demeaner.use_gpu})")
except Exception as e:
    print(f"   ✗ Instantiation failed: {e}")
    exit(1)

# Test 3: Basic demean operation
print("\n3. Testing basic demean operation...")
try:
    rng = np.random.default_rng(42)
    N = 100
    x = rng.normal(0, 1, N * 2).reshape((N, 2))
    flist = np.column_stack([
        rng.integers(0, 5, N),
        rng.integers(0, 3, N)
    ]).astype(np.uint64)
    weights = np.ones(N)

    x_demeaned, success = demeaner.demean(x, flist, weights, tol=1e-10)

    assert success, "Demeaning should succeed"
    assert x_demeaned.shape == x.shape, "Output shape should match input"
    assert isinstance(x_demeaned, np.ndarray), "Output should be numpy array"
    print(f"   ✓ Demeaning successful (shape: {x_demeaned.shape})")
except Exception as e:
    print(f"   ✗ Demean operation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Functional interface (legacy flist)
print("\n4. Testing functional interface with flist...")
try:
    x_demeaned2, success2 = demean_cupy(x, flist=flist, weights=weights, tol=1e-10)

    assert success2, "Functional interface should succeed"
    assert x_demeaned2.shape == x.shape, "Output shape should match input"
    print(f"   ✓ Functional interface works (flist)")
except Exception as e:
    print(f"   ✗ Functional interface failed: {e}")
    exit(1)

# Test 5: Backend integration
print("\n5. Testing backend integration...")
try:
    from pyfixest.estimation.backends import BACKENDS

    assert "cupy" in BACKENDS, "cupy backend should be in BACKENDS dict"
    cupy_backend = BACKENDS["cupy"]
    assert "demean" in cupy_backend, "cupy backend should have demean function"
    print(f"   ✓ Backend integration successful")
except Exception as e:
    print(f"   ✗ Backend integration failed: {e}")
    exit(1)

# Test 6: Backend selector
print("\n6. Testing backend selector...")
try:
    from pyfixest.estimation.demean_ import _set_demeaner_backend

    demean_func = _set_demeaner_backend("cupy")
    if CUPY_AVAILABLE:
        assert demean_func == demean_cupy, "Should return demean_cupy function"
    print(f"   ✓ Backend selector works")
except Exception as e:
    print(f"   ✗ Backend selector failed: {e}")
    exit(1)

# Test 7: Sparse matrix creation (legacy internal method)
print("\n7. Testing internal sparse matrix creation...")
try:
    D = demeaner._create_fe_sparse_matrix(flist)

    expected_cols = int(flist[:, 0].max()) + 1 + int(flist[:, 1].max()) + 1
    assert D.shape == (N, expected_cols), f"Expected shape (100, {expected_cols}), got {D.shape}"
    print(f"   ✓ Internal sparse matrix created (shape: {D.shape}, nnz: {D.nnz})")
except Exception as e:
    print(f"   ✗ Sparse matrix creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 8: Formulaic sparse matrix creation
if FORMULAIC_AVAILABLE:
    print("\n8. Testing formulaic sparse matrix creation...")
    try:
        # Create FE DataFrame
        fe = pd.DataFrame({
            "fe1": flist[:, 0],
            "fe2": flist[:, 1]
        })

        # Test with drop_reference=True (default)
        D_formulaic = create_fe_sparse_matrix(fe, drop_reference=True)
        print(f"   ✓ Formulaic matrix created (drop_reference=True)")
        print(f"     Shape: {D_formulaic.shape}, nnz: {D_formulaic.nnz}")
        print(f"     Density: {D_formulaic.nnz / (D_formulaic.shape[0] * D_formulaic.shape[1]):.2%}")

        # Test with drop_reference=False
        D_formulaic_full = create_fe_sparse_matrix(fe, drop_reference=False)
        print(f"   ✓ Formulaic matrix created (drop_reference=False)")
        print(f"     Shape: {D_formulaic_full.shape}, nnz: {D_formulaic_full.nnz}")

        assert D_formulaic_full.shape[1] > D_formulaic.shape[1], \
            "Full rank matrix should have more columns"
    except Exception as e:
        print(f"   ✗ Formulaic sparse matrix creation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print("\n8. Skipping formulaic tests (not available)")

# Test 9: New functional interface with fe DataFrame
if FORMULAIC_AVAILABLE:
    print("\n9. Testing functional interface with fe DataFrame...")
    try:
        fe = pd.DataFrame({
            "fe1": flist[:, 0],
            "fe2": flist[:, 1]
        })

        x_demeaned_fe, success_fe = demean_cupy(x, fe=fe, weights=weights, tol=1e-10)

        assert success_fe, "Functional interface with fe should succeed"
        assert x_demeaned_fe.shape == x.shape, "Output shape should match input"

        # Compare with flist approach
        diff = np.abs(x_demeaned_fe - x_demeaned2).max()
        print(f"   ✓ Functional interface works (fe DataFrame)")
        print(f"     Max diff vs flist approach: {diff:.2e}")
    except Exception as e:
        print(f"   ✗ Functional interface with fe failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print("\n9. Skipping fe DataFrame tests (formulaic not available)")

# Test 10: Compare with Rust backend
print("\n10. Comparing with Rust backend...")
try:
    from pyfixest.core import demean as demean_rs

    x_demeaned_rust, success_rust = demean_rs(x, flist, weights, tol=1e-10)

    assert success_rust, "Rust backend should succeed"

    # Show sample values
    print(f"\n   Sample values (first 5 observations, first column):")
    print(f"   Rust backend:         {x_demeaned_rust[:5, 0]}")
    print(f"   CuPy (flist):         {x_demeaned2[:5, 0]}")
    if FORMULAIC_AVAILABLE:
        print(f"   CuPy (fe DataFrame):  {x_demeaned_fe[:5, 0]}")

    # Compare flist approach
    diff_flist = np.abs(x_demeaned2 - x_demeaned_rust).max()
    print(f"\n   ✓ Rust comparison (flist): max diff = {diff_flist:.2e}")

    # Compare fe approach if available
    if FORMULAIC_AVAILABLE:
        diff_fe = np.abs(x_demeaned_fe - x_demeaned_rust).max()
        print(f"   ✓ Rust comparison (fe):    max diff = {diff_fe:.2e}")

        if diff_fe < diff_flist:
            improvement = diff_flist / diff_fe
            print(f"     → Formulaic approach is {improvement:.1f}x more accurate!")

except Exception as e:
    print(f"   ✗ Rust comparison failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
