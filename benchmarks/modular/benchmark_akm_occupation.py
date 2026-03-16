from __future__ import annotations

import sys
from pathlib import Path

from dgps import get_akm_occupation_scenarios
from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
)
from interfaces import FeolsSpec
from runner import export_and_plot, run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_ITERS = 3
BURN_IN = 1
DEFAULT_N_OBS = 1_000_000
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data" / "akm_occupation"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_akm_occupation.csv"
FIGURE_DIR = PROJECT_ROOT / "docs" / "explanation" / "figures" / "akm-occupation-benchmarks"

DGPS = get_akm_occupation_scenarios(DATA_DIR)

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=["x1"],
        fe_cols=["indiv_id", "firm_id", "year", "occ_id"],
        vcov="iid",
    ),
]

BENCHMARKERS = [
    FixestFeolsBenchmarker("fixest-map"),
    JuliaFeolsBenchmarker("FEM.jl (lsmr)"),
    PyFeolsBenchmarkerFullApi("pyfixest (rust-cg)", "rust-cg", fixef_maxiter=10000),
    PyFeolsBenchmarkerFullApi("pyfixest (rust-map)", "rust", fixef_maxiter=10000),
]


def generate_akm_occupation_datasets():
    datasets = []
    for dgp in DGPS:
        print(f"[data] generating {dgp.dgp_name} n={DEFAULT_N_OBS:,}")
        datasets.extend(dgp.generate(n=DEFAULT_N_OBS, n_iters=N_ITERS, burn_in=BURN_IN))
    print(f"[data] {len(datasets)} datasets ready")
    return datasets


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_akm_occupation_datasets()
    results = run_benchmarks(BENCHMARKERS, datasets, SPECS)
    export_and_plot(results, OUTPUT_CSV, figure_dir=FIGURE_DIR)
