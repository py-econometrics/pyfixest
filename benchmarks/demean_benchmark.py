#!/usr/bin/env python3
"""
Benchmark script for comparing demeaning implementations.

Oriented on fixest_benchmarks/bench_ols.R but focused on demeaning only
and optimized for fast iteration.

Usage:
    python benchmarks/demean_benchmark.py           # Fast mode (~30s)
    python benchmarks/demean_benchmark.py --full    # Full mode (~5min)
    python benchmarks/demean_benchmark.py --save    # Save results to JSON
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import numpy as np


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    n_obs: int
    dgp_type: str  # "simple" or "difficult"
    n_fe: int
    n_iters: int


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    config: BenchmarkConfig
    backend: str
    times: list[float]
    median_time: float
    available: bool
    error: str | None = None


def generate_dgp(
    n: int,
    dgp_type: str = "simple",
    n_years: int = 10,
    n_indiv_per_firm: int = 23,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data matching fixest_benchmarks DGP.

    Parameters
    ----------
    n : int
        Number of observations
    dgp_type : str
        "simple" (random firm assignment) or "difficult" (sequential)
    n_years : int
        Number of years
    n_indiv_per_firm : int
        Average individuals per firm

    Returns
    -------
    x : np.ndarray
        Feature matrix (n, 1)
    flist : np.ndarray
        Fixed effect IDs (n, 2 or 3) - [indiv_id, year] or [indiv_id, year, firm_id]
    weights : np.ndarray
        Sample weights (n,)
    """
    n_indiv = max(1, round(n / n_years))
    n_firm = max(1, round(n_indiv / n_indiv_per_firm))

    # Create FE IDs
    indiv_id = np.repeat(np.arange(n_indiv), n_years)[:n]
    year = np.tile(np.arange(n_years), n_indiv)[:n]

    if dgp_type == "simple":
        # Random firm assignment - easier convergence
        firm_id = np.random.randint(0, n_firm, size=n)
    elif dgp_type == "difficult":
        # Sequential firm assignment - harder convergence (messy data)
        firm_id = np.tile(np.arange(n_firm), (n // n_firm) + 1)[:n]
    else:
        raise ValueError(f"Unknown dgp_type: {dgp_type}")

    # Generate features
    x1 = np.random.randn(n)

    # Generate y with FE structure
    firm_fe = np.random.randn(n_firm)[firm_id]
    unit_fe = np.random.randn(n_indiv)[indiv_id]
    year_fe = np.random.randn(n_years)[year]
    y = x1 + firm_fe + unit_fe + year_fe + np.random.randn(n)

    # Stack into matrices
    x = np.column_stack([y, x1])  # Demean both y and x1
    weights = np.ones(n)

    return x, indiv_id, year, firm_id, weights


def get_demean_backends() -> dict[str, Callable | None]:
    """Get available demeaning backends with graceful fallbacks."""
    backends: dict[str, Callable | None] = {}

    # Rust accelerated (default)
    try:
        from pyfixest.core.demean import demean as demean_rust

        backends["rust-accelerated"] = demean_rust
    except ImportError:
        backends["rust-accelerated"] = None

    # Rust simple (via env var)
    def demean_rust_simple(x, flist, weights, tol=1e-8, maxiter=100_000):
        os.environ["PYFIXEST_DEMEAN_SIMPLE"] = "1"
        try:
            from pyfixest.core.demean import demean as demean_rust

            return demean_rust(x, flist, weights, tol, maxiter)
        finally:
            del os.environ["PYFIXEST_DEMEAN_SIMPLE"]

    backends["rust-simple"] = (
        demean_rust_simple if backends["rust-accelerated"] else None
    )

    # Numba
    try:
        from pyfixest.estimation.demean_ import demean as demean_numba

        backends["numba"] = demean_numba
    except ImportError:
        backends["numba"] = None

    # CuPy 32-bit
    try:
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy32

        backends["cupy32"] = demean_cupy32
    except ImportError:
        backends["cupy32"] = None

    # CuPy 64-bit
    try:
        from pyfixest.estimation.cupy.demean_cupy_ import demean_cupy64

        backends["cupy64"] = demean_cupy64
    except ImportError:
        backends["cupy64"] = None

    # R fixest via rpy2 - use feols with only FE (no covariates) to measure demean time
    try:
        import pandas as pd
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()
        importr("fixest")  # Load fixest package

        def demean_fixest(x, flist, weights, tol=1e-8, maxiter=100_000):
            # Create a minimal regression problem that exercises the demeaning
            _n, k = x.shape
            n_fe = flist.shape[1] if flist.ndim > 1 else 1

            # Build a dataframe with y and FE columns
            data = {"y": x[:, 0]}
            fe_names = []
            for j in range(n_fe):
                fe_col = f"fe{j + 1}"
                fe_names.append(fe_col)
                if flist.ndim > 1:
                    data[fe_col] = flist[:, j].astype(int)
                else:
                    data[fe_col] = flist.astype(int)

            df = pd.DataFrame(data)
            r_df = pandas2ri.py2rpy(df)

            # Build formula: y ~ 1 | fe1 + fe2 + ...
            fe_formula = " + ".join(fe_names)
            formula = f"y ~ 1 | {fe_formula}"

            # Call feols (this includes demeaning time)
            ro.r.assign("df", r_df)
            ro.r(f"result <- fixest::feols({formula}, data=df, nthreads=1)")

            # Return the residuals as "demeaned" values
            resid = np.array(ro.r("residuals(result)"))
            result = np.column_stack([resid] + [x[:, j] for j in range(1, k)])
            return result, True

        backends["fixest"] = demean_fixest
    except (ImportError, Exception):
        backends["fixest"] = None

    return backends


def run_single_benchmark(
    demean_func: Callable,
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    n_iters: int,
) -> list[float]:
    """Run a single benchmark configuration multiple times."""
    times = []

    for _ in range(n_iters):
        # Copy arrays to avoid caching effects
        x_copy = x.copy()

        start = time.perf_counter()
        demean_func(x_copy, flist, weights)
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return times


def run_benchmarks(
    configs: list[BenchmarkConfig],
    backends: dict[str, Callable | None],
) -> list[BenchmarkResult]:
    """Run all benchmark configurations across all backends."""
    results = []

    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Config: n={config.n_obs:,}, type={config.dgp_type}, fe={config.n_fe}")
        print("=" * 60)

        # Generate data
        x, indiv_id, year, firm_id, weights = generate_dgp(
            config.n_obs, config.dgp_type
        )

        # Build flist based on n_fe
        if config.n_fe == 2:
            flist = np.column_stack([indiv_id, year]).astype(np.uint64)
        else:  # n_fe == 3
            flist = np.column_stack([indiv_id, year, firm_id]).astype(np.uint64)

        for backend_name, demean_func in backends.items():
            if demean_func is None:
                result = BenchmarkResult(
                    config=config,
                    backend=backend_name,
                    times=[],
                    median_time=float("inf"),
                    available=False,
                    error="Not installed",
                )
                results.append(result)
                print(f"  {backend_name:20s}: not available")
                continue

            try:
                times = run_single_benchmark(
                    demean_func, x, flist, weights, config.n_iters
                )
                med_time = median(times)
                result = BenchmarkResult(
                    config=config,
                    backend=backend_name,
                    times=times,
                    median_time=med_time,
                    available=True,
                )
                results.append(result)
                print(
                    f"  {backend_name:20s}: {med_time * 1000:8.2f} ms (median of {len(times)})"
                )
            except Exception as e:
                result = BenchmarkResult(
                    config=config,
                    backend=backend_name,
                    times=[],
                    median_time=float("inf"),
                    available=False,
                    error=str(e),
                )
                results.append(result)
                print(f"  {backend_name:20s}: ERROR - {e}")

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by config
    configs = sorted(
        set((r.config.n_obs, r.config.dgp_type, r.config.n_fe) for r in results)
    )

    backends = sorted(set(r.backend for r in results))

    # Header
    header = f"{'Config':30s}"
    for backend in backends:
        header += f" {backend:>12s}"
    print(header)
    print("-" * len(header))

    # Find fixest baseline for relative comparison
    fixest_times = {}
    for r in results:
        if r.backend == "fixest" and r.available:
            key = (r.config.n_obs, r.config.dgp_type, r.config.n_fe)
            fixest_times[key] = r.median_time

    # Rows
    for n_obs, dgp_type, n_fe in configs:
        config_str = f"n={n_obs:,} {dgp_type:9s} {n_fe}FE"
        row = f"{config_str:30s}"

        key = (n_obs, dgp_type, n_fe)
        baseline = fixest_times.get(key)

        for backend in backends:
            matching = [
                r
                for r in results
                if r.config.n_obs == n_obs
                and r.config.dgp_type == dgp_type
                and r.config.n_fe == n_fe
                and r.backend == backend
            ]
            if matching and matching[0].available:
                time_ms = matching[0].median_time * 1000
                if baseline and backend != "fixest":
                    ratio = matching[0].median_time / baseline
                    row += f" {time_ms:7.1f}ms({ratio:.1f}x)"
                else:
                    row += f" {time_ms:12.1f}ms"
            else:
                row += f" {'N/A':>12s}"

        print(row)


def save_results(results: list[BenchmarkResult], path: Path) -> None:
    """Save results to JSON."""
    data = []
    for r in results:
        data.append(
            {
                "n_obs": r.config.n_obs,
                "dgp_type": r.config.dgp_type,
                "n_fe": r.config.n_fe,
                "n_iters": r.config.n_iters,
                "backend": r.backend,
                "times": r.times,
                "median_time": r.median_time if r.median_time != float("inf") else None,
                "available": r.available,
                "error": r.error,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


def main():
    """Run demeaning benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark demeaning implementations")
    parser.add_argument(
        "--full", action="store_true", help="Run full benchmark (slower)"
    )
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/benchmark.json"),
        help="Output path for results",
    )
    args = parser.parse_args()

    # Define configurations
    if args.full:
        configs = [
            # Small (fast)
            BenchmarkConfig(n_obs=10_000, dgp_type="simple", n_fe=2, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="difficult", n_fe=2, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="simple", n_fe=3, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="difficult", n_fe=3, n_iters=5),
            # Medium
            BenchmarkConfig(n_obs=100_000, dgp_type="simple", n_fe=2, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="difficult", n_fe=2, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="simple", n_fe=3, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="difficult", n_fe=3, n_iters=3),
            # Large
            BenchmarkConfig(n_obs=500_000, dgp_type="simple", n_fe=2, n_iters=2),
            BenchmarkConfig(n_obs=500_000, dgp_type="difficult", n_fe=2, n_iters=2),
            BenchmarkConfig(n_obs=1_000_000, dgp_type="simple", n_fe=2, n_iters=1),
            BenchmarkConfig(n_obs=1_000_000, dgp_type="difficult", n_fe=2, n_iters=1),
        ]
    else:
        # Fast mode - minimal configs for quick iteration
        configs = [
            BenchmarkConfig(n_obs=10_000, dgp_type="simple", n_fe=2, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="difficult", n_fe=2, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="simple", n_fe=3, n_iters=5),
            BenchmarkConfig(n_obs=10_000, dgp_type="difficult", n_fe=3, n_iters=5),
            BenchmarkConfig(n_obs=100_000, dgp_type="simple", n_fe=2, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="difficult", n_fe=2, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="simple", n_fe=3, n_iters=3),
            BenchmarkConfig(n_obs=100_000, dgp_type="difficult", n_fe=3, n_iters=3),
        ]

    print("Demeaning Benchmark")
    print("=" * 60)
    print(f"Mode: {'full' if args.full else 'fast'}")
    print(f"Configurations: {len(configs)}")

    # Get available backends
    backends = get_demean_backends()
    available = [name for name, func in backends.items() if func is not None]
    unavailable = [name for name, func in backends.items() if func is None]

    print(f"Available backends: {', '.join(available)}")
    if unavailable:
        print(f"Unavailable backends: {', '.join(unavailable)}")

    # Run benchmarks
    results = run_benchmarks(configs, backends)

    # Print summary
    print_summary(results)

    # Save if requested
    if args.save:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
