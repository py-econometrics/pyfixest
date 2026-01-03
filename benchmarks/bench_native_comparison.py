#!/usr/bin/env python3
"""
Benchmark comparing pyfixest feols vs native fixest feols.

Runs fixest directly in R to avoid rpy2 overhead, then compares with pyfixest.
This is a fair apples-to-apples comparison of full feols() routines.
"""

from __future__ import annotations

import os

# Set thread count for Rayon (pyfixest) BEFORE importing pyfixest
os.environ["RAYON_NUM_THREADS"] = "2"

import json
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd


def generate_dgp(
    n: int,
    dgp_type: str = "simple",
    n_years: int = 10,
    n_indiv_per_firm: int = 23,
) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "indiv_id": indiv_id,
            "year": year,
            "firm_id": firm_id,
        }
    )


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


def run_pyfixest_benchmark(
    df: pd.DataFrame,
    n_fe: int,
    n_runs: int = 5,
) -> dict:
    """Run pyfixest feols benchmark."""
    import pyfixest as pf

    # Build formula matching R benchmark
    if n_fe == 2:
        fml = "y ~ x1 | indiv_id + year"
    else:
        fml = "y ~ x1 | indiv_id + year + firm_id"

    # Warmup - use rust backend for accelerated demeaning
    pf.feols(fml, data=df, demeaner_backend="rust")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fit = pf.feols(fml, data=df, demeaner_backend="rust")
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        "median": median(times),
        "times": times,
        "coef": float(fit.coef().iloc[0]),
    }


def main():
    """Run benchmark comparing pyfixest feols vs native fixest feols."""
    configs = [
        (10_000, "simple", 2),
        (10_000, "difficult", 2),
        (10_000, "simple", 3),
        (10_000, "difficult", 3),
        (100_000, "simple", 2),
        (100_000, "difficult", 2),
        (100_000, "simple", 3),
        (100_000, "difficult", 3),
        (1_000_000, "simple", 2),
        (1_000_000, "difficult", 2),
        (1_000_000, "simple", 3),
        (1_000_000, "difficult", 3),
    ]

    results = []

    print("=" * 70)
    print("PyFixest feols() vs Fixest feols() Benchmark")
    print("=" * 70)

    for n_obs, dgp_type, n_fe in configs:
        print(f"\nConfig: n={n_obs:,}, type={dgp_type}, fe={n_fe}")
        print("-" * 50)

        # Generate data
        df = generate_dgp(n_obs, dgp_type)

        # Run R benchmark (feols)
        r_result = run_r_benchmark(n_obs, dgp_type, n_fe)
        r_time = r_result.get("median", float("inf"))
        print(f"  fixest (R):      {r_time:8.2f} ms")

        # Run pyfixest benchmark (feols)
        py_result = run_pyfixest_benchmark(df, n_fe)
        py_time = py_result.get("median", float("inf"))

        if r_time > 0 and py_time < float("inf"):
            ratio = py_time / r_time
            print(f"  pyfixest:        {py_time:8.2f} ms ({ratio:.2f}x)")
        else:
            print(f"  pyfixest:        {py_time:8.2f} ms")

        results.append(
            {
                "n_obs": n_obs,
                "dgp_type": dgp_type,
                "n_fe": n_fe,
                "fixest_r_ms": r_time,
                "pyfixest_ms": py_time,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (pyfixest feols vs fixest feols)")
    print("=" * 70)

    print(f"{'Config':<35} {'fixest':>10} {'pyfixest':>10} {'ratio':>8}")
    print("-" * 65)

    for r in results:
        config = f"n={r['n_obs']:,} {r['dgp_type']:9} {r['n_fe']}FE"
        fixest = r["fixest_r_ms"]
        pyfixest = r["pyfixest_ms"]

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
