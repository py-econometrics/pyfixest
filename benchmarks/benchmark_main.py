from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demeaners import FixestDemeaner, PyFixestDemeaner
from dgps import DifficultDGP, SimpleDGP
from interfaces import BenchmarkSpec
from plotting import plot_benchmarks

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
SIZES = [1_000, 10_000, 100_000]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = Path("benchmarks/data")
OUTPUT_CSV = Path("benchmarks/results/bench.csv")
DEMEANERS = [
    PyFixestDemeaner("numba"),
    PyFixestDemeaner("rust"),
    FixestDemeaner(),
]
SPECS = [
    BenchmarkSpec(demean_cols=["y", "x1"], fe_cols=["indiv_id", "year"]),
    BenchmarkSpec(
        demean_cols=["y", "x1"],
        fe_cols=["indiv_id", "year", "firm_id"],
    ),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    dgps = [SimpleDGP(DATA_DIR), DifficultDGP(DATA_DIR)]

    all_datasets = []
    for dgp in dgps:
        for n in SIZES:
            all_datasets.extend(dgp.generate(n=n, n_iters=N_ITERS, burn_in=BURN_IN))

    all_results = []
    for demeaner in DEMEANERS:
        for spec in SPECS:
            all_results.extend(demeaner.run(all_datasets, spec))

    results_df = pd.DataFrame([asdict(result) for result in all_results])
    results_df = results_df[results_df["iter_type"] != "burnin"].copy()
    results_df.to_csv(OUTPUT_CSV, index=False)

    plot_df = results_df[results_df["success"] & results_df["time"].notna()].copy()
    plot_benchmarks(plot_df, OUTPUT_CSV.with_suffix(".png"))


if __name__ == "__main__":
    main()
