"""
Benchmark: LSMR solver with vs. without diagonal preconditioning.

Generates problems of increasing difficulty and compares iteration count,
wall-clock time, and residual accuracy for the two modes.
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pyfixest.estimation.cupy.demean_cupy_ import (
    CupyFWLDemeaner,
    create_fe_sparse_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem(
    n: int,
    n_x: int,
    fe_spec: dict[str, int],
    *,
    imbalance: float = 0.0,
    seed: int = 0,
):
    """
    Build a demeaning problem.

    Parameters
    ----------
    n : int
        Number of observations.
    n_x : int
        Number of columns in X.
    fe_spec : dict
        Mapping of FE name -> number of levels, e.g. {"f1": 50, "f2": 100}.
    imbalance : float in [0, 1)
        0 = perfectly balanced groups, close to 1 = highly skewed sizes.
        Uses a Dirichlet draw to create unequal group probabilities.
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n_x))
    weights = np.ones(n)

    fe_arrays = {}
    for name, n_levels in fe_spec.items():
        if imbalance > 0:
            # Dirichlet with low concentration -> skewed groups
            alpha = np.full(n_levels, 1.0 - imbalance + 0.01)
            probs = rng.dirichlet(alpha)
            fe_arrays[name] = rng.choice(n_levels, size=n, p=probs).astype(np.uint64)
        else:
            fe_arrays[name] = rng.integers(0, n_levels, size=n).astype(np.uint64)

    fe_df = pd.DataFrame(fe_arrays)
    flist = fe_df.values
    D = create_fe_sparse_matrix(fe_df)

    return x, flist, weights, D


def _run_once(demeaner, x, flist, weights, D):
    """Time a single demean call and return (elapsed, success, residual_norm)."""
    start = time.perf_counter()
    x_dm, success = demeaner.demean(
        x, flist, weights, fe_sparse_matrix=D
    )
    elapsed = time.perf_counter() - start
    resid_norm = np.linalg.norm(x_dm)
    return elapsed, success, resid_norm


def _benchmark_scenario(label, n, n_x, fe_spec, imbalance, n_repeats=3):
    """Run both preconditioned and unpreconditioned, print results."""
    x, flist, weights, D = _make_problem(
        n, n_x, fe_spec, imbalance=imbalance
    )

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  n={n:,}  n_x={n_x}  FEs={fe_spec}  imbalance={imbalance}")
    print(f"  D shape: {D.shape}  nnz: {D.nnz:,}")
    print(f"{'=' * 70}")

    for use_precond in [False, True]:
        tag = "preconditioned" if use_precond else "   unprecond.  "
        times = []
        for _ in range(n_repeats):
            demeaner = CupyFWLDemeaner(
                use_gpu=False,
                warn_on_cpu_fallback=False,
                use_preconditioner=use_precond,
            )
            elapsed, success, resid_norm = _run_once(demeaner, x, flist, weights, D)
            times.append(elapsed)

        med = np.median(times)
        print(
            f"  [{tag}]  median={med:.4f}s  "
            f"success={success}  ||residual||={resid_norm:.6e}"
        )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def main():
    print("LSMR Preconditioner Benchmark")
    print("SciPy CPU-only (use_gpu=False)\n")

    # ---- Easy: balanced, one FE, small ----
    _benchmark_scenario(
        label="Easy: balanced, 1 FE, small",
        n=10_000,
        n_x=3,
        fe_spec={"f1": 50},
        imbalance=0.0,
    )

    # ---- Easy: balanced, one FE, medium ----
    _benchmark_scenario(
        label="Easy: balanced, 1 FE, medium",
        n=100_000,
        n_x=5,
        fe_spec={"f1": 200},
        imbalance=0.0,
    )

    # ---- Moderate: two FEs, balanced ----
    _benchmark_scenario(
        label="Moderate: 2 balanced FEs",
        n=100_000,
        n_x=5,
        fe_spec={"f1": 200, "f2": 300},
        imbalance=0.0,
    )

    # ---- Moderate: one FE, mildly imbalanced ----
    _benchmark_scenario(
        label="Moderate: 1 FE, mild imbalance",
        n=100_000,
        n_x=5,
        fe_spec={"f1": 500},
        imbalance=0.5,
    )

    # ---- Hard: two FEs, highly imbalanced ----
    _benchmark_scenario(
        label="Hard: 2 FEs, high imbalance",
        n=200_000,
        n_x=5,
        fe_spec={"f1": 1_000, "f2": 500},
        imbalance=0.9,
    )

    # ---- Hard: three FEs, highly imbalanced ----
    _benchmark_scenario(
        label="Hard: 3 FEs, high imbalance",
        n=200_000,
        n_x=5,
        fe_spec={"f1": 500, "f2": 300, "f3": 200},
        imbalance=0.9,
    )

    # ---- Very hard: high-dimensional FE ----
    _benchmark_scenario(
        label="Very hard: high-dim FE (5000 levels), imbalanced",
        n=300_000,
        n_x=3,
        fe_spec={"f1": 5_000},
        imbalance=0.9,
    )

    # ---- Very hard: two high-dim FEs ----
    _benchmark_scenario(
        label="Very hard: 2 high-dim FEs, imbalanced",
        n=500_000,
        n_x=3,
        fe_spec={"f1": 5_000, "f2": 2_000},
        imbalance=0.9,
    )


if __name__ == "__main__":
    main()
