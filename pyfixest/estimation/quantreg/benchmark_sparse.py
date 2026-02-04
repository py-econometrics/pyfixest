"""
Benchmark script comparing original, optimized, and sparse Frisch-Newton solvers.

This script benchmarks the performance of three solver implementations:
1. Original dense solver (frisch_newton_ip.py) - current production solver
2. Optimized dense solver (frisch_newton_optimized.py) - Cholesky reuse optimization
3. Sparse solver (frisch_newton_sparse.py) - sparse matrix support

Usage:
    python -m pyfixest.estimation.quantreg.benchmark_sparse

Output:
    - Timing comparisons for different problem configurations
    - Speedup ratios
    - Verification that all methods produce identical results
"""

import time
from typing import Callable

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, solve_triangular
from scipy.sparse import csr_matrix

# Import all three implementations
from pyfixest.estimation.quantreg.frisch_newton_ip import (
    frisch_newton_solver as frisch_newton_original,
)
from pyfixest.estimation.quantreg.frisch_newton_optimized import (
    frisch_newton_optimized,
)
from pyfixest.estimation.quantreg.frisch_newton_sparse import (
    frisch_newton_solver_sparse,
)


def create_design_matrix_with_categoricals(
    n_obs: int,
    n_continuous: int,
    categorical_levels: list[int],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Create a design matrix with continuous variables and categorical dummies.

    Parameters
    ----------
    n_obs : int
        Number of observations
    n_continuous : int
        Number of continuous covariates
    categorical_levels : list[int]
        Number of levels for each categorical variable (dummies = levels - 1)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        - X : np.ndarray - Design matrix (n_obs, k)
        - Y : np.ndarray - Response variable (n_obs, 1)
        - sparsity : float - Fraction of zeros in X
    """
    rng = np.random.default_rng(seed)

    # Intercept
    columns = [np.ones((n_obs, 1))]

    # Continuous variables
    for _ in range(n_continuous):
        columns.append(rng.standard_normal((n_obs, 1)))

    # Categorical dummies (reference category dropped)
    for n_levels in categorical_levels:
        cat_values = rng.integers(0, n_levels, size=n_obs)
        # Create dummies for levels 1 to n_levels-1 (0 is reference)
        dummies = np.zeros((n_obs, n_levels - 1))
        for level in range(1, n_levels):
            dummies[:, level - 1] = (cat_values == level).astype(float)
        columns.append(dummies)

    X = np.hstack(columns)
    k = X.shape[1]

    # Generate Y with some relationship to X
    true_beta = rng.standard_normal(k)
    Y = X @ true_beta + rng.standard_normal(n_obs)
    Y = Y.reshape(-1, 1)

    # Compute sparsity
    sparsity = 1.0 - np.count_nonzero(X) / X.size

    return X, Y, sparsity


def prepare_lp_inputs(
    X: np.ndarray, Y: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare inputs for the LP formulation of quantile regression.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (N, k)
    Y : np.ndarray
        Response variable (N, 1)
    q : float
        Quantile level

    Returns
    -------
    tuple
        A, b, c, u for the LP formulation
    """
    N = X.shape[0]
    A = X.T  # (k, N)
    b = (1 - q) * X.T @ np.ones(N)
    c = -Y.ravel()
    u = np.ones(N)
    return A, b, c, u


def run_original_solver(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Run original dense solver with timing.

    Returns
    -------
    tuple
        beta_hat, converged, iterations, elapsed_time
    """
    # Prepare chol and P as in the original code
    X = A.T  # (N, k)
    _chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
    _chol = np.atleast_2d(_chol)
    _P = solve_triangular(_chol, X.T, lower=True, check_finite=False)

    start = time.perf_counter()
    result = frisch_newton_original(
        A=A,
        b=b,
        c=c,
        u=u,
        q=q,
        tol=tol,
        max_iter=max_iter,
        chol=_chol,
        P=_P,
        backoff=0.9995,
    )
    elapsed = time.perf_counter() - start

    return result[0], result[1], result[2], elapsed


def run_optimized_solver(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Run optimized dense solver (Cholesky reuse) with timing.

    Returns
    -------
    tuple
        beta_hat, converged, iterations, elapsed_time
    """
    start = time.perf_counter()
    result = frisch_newton_optimized(
        A=A,
        b=b,
        c=c,
        u=u,
        q=q,
        tol=tol,
        max_iter=max_iter,
        backoff=0.9995,
    )
    elapsed = time.perf_counter() - start

    return result[0], result[1], result[2], elapsed


def run_sparse_solver_dense_input(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Run sparse solver with dense input (for comparison).

    Returns
    -------
    tuple
        beta_hat, converged, iterations, elapsed_time
    """
    start = time.perf_counter()
    result = frisch_newton_solver_sparse(
        A=A,
        b=b,
        c=c,
        u=u,
        q=q,
        tol=tol,
        max_iter=max_iter,
        backoff=0.9995,
    )
    elapsed = time.perf_counter() - start

    return result[0], result[1], result[2], elapsed


def run_sparse_solver_sparse_input(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    q: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, bool, int, float]:
    """
    Run sparse solver with sparse matrix input.

    Returns
    -------
    tuple
        beta_hat, converged, iterations, elapsed_time
    """
    A_sparse = csr_matrix(A)

    start = time.perf_counter()
    result = frisch_newton_solver_sparse(
        A=A_sparse,
        b=b,
        c=c,
        u=u,
        q=q,
        tol=tol,
        max_iter=max_iter,
        backoff=0.9995,
    )
    elapsed = time.perf_counter() - start

    return result[0], result[1], result[2], elapsed


def benchmark_single_config(
    n_obs: int,
    n_continuous: int,
    categorical_levels: list[int],
    q: float = 0.5,
    tol: float = 1e-6,
    n_runs: int = 3,
    seed: int = 42,
) -> dict:
    """
    Benchmark all solvers for a single configuration.

    Parameters
    ----------
    n_obs : int
        Number of observations
    n_continuous : int
        Number of continuous variables
    categorical_levels : list[int]
        Levels for each categorical variable
    q : float
        Quantile level
    tol : float
        Convergence tolerance
    n_runs : int
        Number of timing runs (best of n)
    seed : int
        Random seed

    Returns
    -------
    dict
        Benchmark results including timings and verification
    """
    # Create test data
    X, Y, sparsity = create_design_matrix_with_categoricals(
        n_obs, n_continuous, categorical_levels, seed
    )
    N, k = X.shape
    max_iter = N

    # Prepare LP inputs
    A, b, c, u = prepare_lp_inputs(X, Y, q)

    results = {
        "n_obs": n_obs,
        "n_coef": k,
        "sparsity": sparsity,
        "n_categorical": len(categorical_levels),
    }

    # Run original solver (skip if pdb.set_trace is still there)
    try:
        times_original = []
        for _ in range(n_runs):
            beta_orig, conv_orig, it_orig, t = run_original_solver(
                A, b, c, u, q, tol, max_iter
            )
            times_original.append(t)
        results["time_original"] = min(times_original)
        results["converged_original"] = conv_orig
        results["iter_original"] = it_orig
        results["beta_original"] = beta_orig
    except Exception as e:
        print(f"  Original solver failed: {e}")
        results["time_original"] = np.nan
        results["beta_original"] = None

    # Run optimized solver
    try:
        times_optimized = []
        for _ in range(n_runs):
            beta_opt, conv_opt, it_opt, t = run_optimized_solver(
                A, b, c, u, q, tol, max_iter
            )
            times_optimized.append(t)
        results["time_optimized"] = min(times_optimized)
        results["converged_optimized"] = conv_opt
        results["iter_optimized"] = it_opt
        results["beta_optimized"] = beta_opt
    except Exception as e:
        print(f"  Optimized solver failed: {e}")
        results["time_optimized"] = np.nan
        results["beta_optimized"] = None

    # Run sparse solver with dense input
    times_sparse_dense = []
    for _ in range(n_runs):
        beta_sd, conv_sd, it_sd, t = run_sparse_solver_dense_input(
            A, b, c, u, q, tol, max_iter
        )
        times_sparse_dense.append(t)
    results["time_sparse_dense"] = min(times_sparse_dense)
    results["converged_sparse_dense"] = conv_sd
    results["iter_sparse_dense"] = it_sd
    results["beta_sparse_dense"] = beta_sd

    # Run sparse solver with sparse input
    times_sparse_sparse = []
    for _ in range(n_runs):
        beta_ss, conv_ss, it_ss, t = run_sparse_solver_sparse_input(
            A, b, c, u, q, tol, max_iter
        )
        times_sparse_sparse.append(t)
    results["time_sparse_sparse"] = min(times_sparse_sparse)
    results["converged_sparse_sparse"] = conv_ss
    results["iter_sparse_sparse"] = it_ss
    results["beta_sparse_sparse"] = beta_ss

    # Compute speedups (relative to original)
    if not np.isnan(results["time_original"]):
        results["speedup_optimized"] = (
            results["time_original"] / results["time_optimized"]
            if not np.isnan(results.get("time_optimized", np.nan))
            else np.nan
        )
        results["speedup_sparse_dense"] = (
            results["time_original"] / results["time_sparse_dense"]
        )
        results["speedup_sparse_sparse"] = (
            results["time_original"] / results["time_sparse_sparse"]
        )
    else:
        results["speedup_optimized"] = np.nan
        results["speedup_sparse_dense"] = np.nan
        results["speedup_sparse_sparse"] = np.nan

    # Verify results match
    all_match = True
    if results["beta_original"] is not None:
        diff_opt = (
            np.max(np.abs(beta_orig - beta_opt))
            if results.get("beta_optimized") is not None
            else np.nan
        )
        diff_dense = np.max(np.abs(beta_orig - beta_sd))
        diff_sparse = np.max(np.abs(beta_orig - beta_ss))
        results["max_diff_optimized"] = diff_opt
        results["max_diff_dense"] = diff_dense
        results["max_diff_sparse"] = diff_sparse
        all_match = diff_dense < 1e-6 and diff_sparse < 1e-6
        if not np.isnan(diff_opt):
            all_match = all_match and diff_opt < 1e-6
        results["results_match"] = all_match
    else:
        # Compare sparse implementations only
        diff = np.max(np.abs(beta_sd - beta_ss))
        results["max_diff_sparse_vs_dense"] = diff
        results["results_match"] = diff < 1e-6

    return results


def run_full_benchmark():
    """
    Run comprehensive benchmark across multiple configurations.
    """
    print("=" * 80)
    print("FRISCH-NEWTON SOLVER BENCHMARK: Original vs Optimized vs Sparse")
    print("=" * 80)
    print()

    # Define test configurations
    configs = [
        # Small problem, low sparsity
        {"n_obs": 500, "n_continuous": 3, "categorical_levels": [2, 3]},
        # Medium problem, moderate sparsity
        {"n_obs": 2000, "n_continuous": 2, "categorical_levels": [2, 4, 4, 6]},
        # Large problem, high sparsity (like debug.py)
        {"n_obs": 5000, "n_continuous": 2, "categorical_levels": [2, 4, 4, 6, 10]},
        # Very large problem
        {"n_obs": 10000, "n_continuous": 2, "categorical_levels": [2, 4, 4, 6, 10, 20]},
        # Extreme sparsity
        {
            "n_obs": 20000,
            "n_continuous": 2,
            "categorical_levels": [2, 5, 5, 10, 10, 20, 50],
        },
    ]

    all_results = []

    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}/{len(configs)}:")
        print(f"  N={config['n_obs']}, continuous={config['n_continuous']}, "
              f"categorical levels={config['categorical_levels']}")

        try:
            result = benchmark_single_config(**config)
            all_results.append(result)

            print(f"  Sparsity: {result['sparsity']:.1%}")
            print(f"  Coefficients: {result['n_coef']}")
            print()
            print(f"  Timing Results (best of 3 runs):")
            if not np.isnan(result.get("time_original", np.nan)):
                print(f"    Original:       {result['time_original']*1000:8.2f} ms")
            if not np.isnan(result.get("time_optimized", np.nan)):
                print(f"    Optimized:      {result['time_optimized']*1000:8.2f} ms")
            print(f"    Sparse(dense):  {result['time_sparse_dense']*1000:8.2f} ms")
            print(f"    Sparse(sparse): {result['time_sparse_sparse']*1000:8.2f} ms")
            print()
            print(f"  Speedup vs Original:")
            if not np.isnan(result.get("speedup_optimized", np.nan)):
                print(f"    Optimized:      {result['speedup_optimized']:.2f}x")
            if not np.isnan(result.get("speedup_sparse_sparse", np.nan)):
                print(f"    Sparse(sparse): {result['speedup_sparse_sparse']:.2f}x")
            print(f"  Results match: {result['results_match']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    summary_data = []
    for r in all_results:
        row = {
            "N": r["n_obs"],
            "k": r["n_coef"],
            "Sparsity": f"{r['sparsity']:.1%}",
        }
        if not np.isnan(r.get("time_original", np.nan)):
            row["T_original (ms)"] = f"{r['time_original']*1000:.1f}"
        if not np.isnan(r.get("time_optimized", np.nan)):
            row["T_optimized (ms)"] = f"{r['time_optimized']*1000:.1f}"
        row["T_sparse_dense (ms)"] = f"{r['time_sparse_dense']*1000:.1f}"
        row["T_sparse_sparse (ms)"] = f"{r['time_sparse_sparse']*1000:.1f}"
        if not np.isnan(r.get("speedup_optimized", np.nan)):
            row["Speedup_opt"] = f"{r['speedup_optimized']:.2f}x"
        else:
            row["Speedup_opt"] = "N/A"
        if not np.isnan(r.get("speedup_sparse_sparse", np.nan)):
            row["Speedup_sparse"] = f"{r['speedup_sparse_sparse']:.2f}x"
        else:
            row["Speedup_sparse"] = "N/A"
        row["Match"] = "Yes" if r["results_match"] else "NO"
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    return all_results


def quick_test():
    """
    Quick test to verify both implementations produce same results.
    """
    print("Quick correctness test...")

    # Simple test case
    X, Y, sparsity = create_design_matrix_with_categoricals(
        n_obs=1000,
        n_continuous=2,
        categorical_levels=[2, 3, 4],
        seed=123,
    )
    q = 0.5
    N, k = X.shape
    A, b, c, u = prepare_lp_inputs(X, Y, q)

    print(f"  Problem size: N={N}, k={k}, sparsity={sparsity:.1%}")

    # Run original solver
    beta_orig, conv0, it0, _ = run_original_solver(A, b, c, u, q, 1e-6, N)
    print(f"  Original:       converged={conv0}, iter={it0}")

    # Run optimized solver
    beta_opt, conv1, it1, _ = run_optimized_solver(A, b, c, u, q, 1e-6, N)
    print(f"  Optimized:      converged={conv1}, iter={it1}")

    # Run sparse solver with dense input
    beta_sparse_d, conv2, it2, _ = run_sparse_solver_dense_input(
        A, b, c, u, q, 1e-6, N
    )
    print(f"  Sparse(dense):  converged={conv2}, iter={it2}")

    # Run sparse solver with sparse input
    beta_sparse_s, conv3, it3, _ = run_sparse_solver_sparse_input(
        A, b, c, u, q, 1e-6, N
    )
    print(f"  Sparse(sparse): converged={conv3}, iter={it3}")

    # Compare all against original
    diff_opt = np.max(np.abs(beta_orig - beta_opt))
    diff_sparse_d = np.max(np.abs(beta_orig - beta_sparse_d))
    diff_sparse_s = np.max(np.abs(beta_orig - beta_sparse_s))
    print(f"  Max diff (optimized vs original):      {diff_opt:.2e}")
    print(f"  Max diff (sparse_dense vs original):   {diff_sparse_d:.2e}")
    print(f"  Max diff (sparse_sparse vs original):  {diff_sparse_s:.2e}")

    all_pass = diff_opt < 1e-6 and diff_sparse_d < 1e-6 and diff_sparse_s < 1e-6
    print(f"  Test {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


if __name__ == "__main__":
    # Run quick test first
    if quick_test():
        print("\n")
        # Run full benchmark
        run_full_benchmark()
    else:
        print("Quick test failed, skipping full benchmark")
