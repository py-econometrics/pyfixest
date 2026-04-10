from __future__ import annotations

import sys
from pathlib import Path

from benchmarker_sets import build_standard_feols_benchmarkers
from dgps import BaseDGP
from interfaces import FeolsSpec
from runner import generate_datasets, plot_results, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
K_VALUES = [1, 5, 10]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_bench.csv"
FIGURE_DIR = PROJECT_ROOT / "docs" / "explanation" / "figures" / "base-benchmarks"

DGPS = [
    BaseDGP(DATA_DIR, "simple", k_values=tuple(K_VALUES)),
    BaseDGP(DATA_DIR, "difficult", k_values=tuple(K_VALUES)),
]

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=[f"x{i}" for i in range(1, k + 1)],
        fe_cols=fe_cols,
        vcov="iid",
    )
    for k in K_VALUES
    for fe_cols in (["indiv_id", "year"], ["indiv_id", "year", "firm_id"])
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DGPS, SIZES, N_ITERS, BURN_IN)
    bundle = build_standard_feols_benchmarkers()
    results_df = run_benchmarks(bundle.benchmarkers, datasets, SPECS, OUTPUT_CSV)
    plot_results(
        results_df,
        OUTPUT_CSV,
        figure_dir=FIGURE_DIR,
        figure_backends=bundle.figure_backends,
    )
