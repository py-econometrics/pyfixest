#!/usr/bin/env python3
"""
Benchmark comparing pyfixest demean vs native fixest (via R subprocess).

Runs fixest directly in R to avoid rpy2 overhead, then compares with pyfixest.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np


def generate_dgp(
    n: int,
    dgp_type: str = "simple",
    n_years: int = 10,
    n_indiv_per_firm: int = 23,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate test data matching fixest benchmark DGP."""
    np.random.seed(42)

    n_indiv = max(1, round(n / n_years))
    n_firm = max(1, round(n_indiv / n_indiv_per_firm))

    indiv_id = np.repeat(np.arange(n_indiv), n_years)[:n]
    year = np.tile(np.arange(n_years), n_indiv)[:n]

    if dgp_type == "simple":
        firm_id = np.random.randint(0, n_firm, size=n)
    else:  # difficult
        firm_id = np.tile(np.arange(n_firm), (n // n_firm) + 1)[:n]

    x1 = np.random.randn(n)
    firm_fe = np.random.randn(n_firm)[firm_id]
    unit_fe = np.random.randn(n_indiv)[indiv_id]
    year_fe = np.random.randn(n_years)[year]
    y = x1 + firm_fe + unit_fe + year_fe + np.random.randn(n)

    x = np.column_stack([y, x1])
    weights = np.ones(n)

    return x, indiv_id, year, firm_id, weights


def run_r_benchmark(n_obs: int, dgp_type: str, n_fe: int, n_runs: int = 5) -> dict:
    """Run fixest benchmark in R subprocess."""
    r_script = Path(__file__).parent / "bench_demean_r.R"

    try:
        result = subprocess.run(
            ["Rscript", str(r_script), str(n_obs), dgp_type, str(n_fe)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {"error": result.stderr, "times": [], "median": float("inf")}

        # Parse output
        lines = result.stdout.strip().split("\n")
        median_ms = None
        for line in lines:
            if "Median:" in line:
                median_ms = float(line.split(":")[1].strip().replace(" ms", ""))

        return {
            "median": median_ms if median_ms else float("inf"),
            "output": result.stdout,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "median": float("inf")}
    except FileNotFoundError:
        return {"error": "R not found", "median": float("inf")}


def run_rust_benchmark(
    x: np.ndarray,
    flist: np.ndarray,
    weights: np.ndarray,
    n_runs: int = 5,
    use_simple: bool = False,
) -> dict:
    """Run pyfixest Rust demean benchmark."""
    import os

    if use_simple:
        os.environ["PYFIXEST_DEMEAN_SIMPLE"] = "1"
    elif "PYFIXEST_DEMEAN_SIMPLE" in os.environ:
        del os.environ["PYFIXEST_DEMEAN_SIMPLE"]

    try:
        from pyfixest.core.demean import demean

        times = []
        for _ in range(n_runs):
            x_copy = x.copy()
            start = time.perf_counter()
            _result, converged = demean(x_copy, flist, weights)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        return {
            "median": median(times),
            "times": times,
            "converged": converged,
        }
    except Exception as e:
        return {"error": str(e), "median": float("inf")}
    finally:
        if "PYFIXEST_DEMEAN_SIMPLE" in os.environ:
            del os.environ["PYFIXEST_DEMEAN_SIMPLE"]


def main():
    """Run benchmark comparing pyfixest demean vs native fixest."""
    configs = [
        (10_000, "simple", 2),
        (10_000, "difficult", 2),
        (10_000, "simple", 3),
        (10_000, "difficult", 3),
        (100_000, "simple", 2),
        (100_000, "difficult", 2),
        (100_000, "simple", 3),
        (100_000, "difficult", 3),
    ]

    results = []

    print("=" * 70)
    print("PyFixest vs Fixest Native Benchmark")
    print("=" * 70)

    for n_obs, dgp_type, n_fe in configs:
        print(f"\nConfig: n={n_obs:,}, type={dgp_type}, fe={n_fe}")
        print("-" * 50)

        # Generate data
        x, indiv_id, year, firm_id, weights = generate_dgp(n_obs, dgp_type)

        if n_fe == 2:
            flist = np.column_stack([indiv_id, year]).astype(np.uint64)
        else:
            flist = np.column_stack([indiv_id, year, firm_id]).astype(np.uint64)

        # Run R benchmark
        r_result = run_r_benchmark(n_obs, dgp_type, n_fe)
        r_time = r_result.get("median", float("inf"))
        print(f"  fixest (R native):   {r_time:8.2f} ms")

        # Run Rust accelerated benchmark
        rust_result = run_rust_benchmark(x, flist, weights)
        rust_time = rust_result.get("median", float("inf"))

        if r_time > 0 and rust_time < float("inf"):
            ratio = rust_time / r_time
            print(f"  pyfixest (Rust):     {rust_time:8.2f} ms ({ratio:.2f}x)")
        else:
            print(f"  pyfixest (Rust):     {rust_time:8.2f} ms")

        # Run Rust simple benchmark
        rust_simple = run_rust_benchmark(x, flist, weights, use_simple=True)
        rust_simple_time = rust_simple.get("median", float("inf"))

        if r_time > 0 and rust_simple_time < float("inf"):
            ratio = rust_simple_time / r_time
            print(f"  pyfixest (simple):   {rust_simple_time:8.2f} ms ({ratio:.2f}x)")
        else:
            print(f"  pyfixest (simple):   {rust_simple_time:8.2f} ms")

        results.append(
            {
                "n_obs": n_obs,
                "dgp_type": dgp_type,
                "n_fe": n_fe,
                "fixest_r_ms": r_time,
                "pyfixest_rust_ms": rust_time,
                "pyfixest_simple_ms": rust_simple_time,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (pyfixest accelerated vs fixest)")
    print("=" * 70)

    print(f"{'Config':<35} {'fixest':>10} {'pyfixest':>10} {'ratio':>8}")
    print("-" * 65)

    for r in results:
        config = f"n={r['n_obs']:,} {r['dgp_type']:9} {r['n_fe']}FE"
        fixest = r["fixest_r_ms"]
        pyfixest = r["pyfixest_rust_ms"]

        if fixest > 0 and fixest < float("inf") and pyfixest < float("inf"):
            ratio = pyfixest / fixest
            print(f"{config:<35} {fixest:>8.1f}ms {pyfixest:>8.1f}ms {ratio:>7.2f}x")
        else:
            print(f"{config:<35} {'N/A':>10} {'N/A':>10}")

    # Save results
    output_path = Path(__file__).parent / "results" / "native_comparison.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
