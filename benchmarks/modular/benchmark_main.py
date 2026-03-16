from __future__ import annotations

import sys
from pathlib import Path

from dgps import BaseDGP
from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
)
from interfaces import FeolsSpec
from runner import export_and_plot, generate_datasets, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_bench.csv"

DGPS = [
    BaseDGP(DATA_DIR, "simple"),
    BaseDGP(DATA_DIR, "difficult"),
]

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=["x1"],
        fe_cols=["indiv_id", "year"],
        vcov="iid",
    ),
    FeolsSpec(
        depvar="y",
        covariates=["x1"],
        fe_cols=["indiv_id", "year", "firm_id"],
        vcov="iid",
    ),
]

BENCHMARKERS = [
    PyFeolsBenchmarkerFullApi("pyfixest (rust-cg)", "rust-cg"),
    PyFeolsBenchmarkerFullApi("pyfixest (rust-map)", "rust"),
    FixestFeolsBenchmarker("fixest-map"),
    JuliaFeolsBenchmarker("FEM.jl (lsmr)"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(DGPS, SIZES, N_ITERS, BURN_IN)
    results = run_benchmarks(BENCHMARKERS, datasets, SPECS)
    export_and_plot(results, OUTPUT_CSV)
