from __future__ import annotations

import sys
from pathlib import Path

from dgps import get_akm_sweep_scenarios
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_ITERS = 3
BURN_IN = 1
DEFAULT_N_OBS = 1_000_000
SCALE_N_OBS = {
    "akm_scale_1": 10_000,
    "akm_scale_2": 100_000,
    "akm_scale_3": 1_000_000,
    "akm_scale_4": 10_000_000,
}
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_akm_sweep.csv"
FIGURE_DIR = PROJECT_ROOT / "docs" / "explanation" / "figures" / "akm-benchmarks"

DGPS = get_akm_sweep_scenarios(DATA_DIR)

SPECS = [
    FeolsSpec(
        depvar="y",
        covariates=["x1"],
        fe_cols=["indiv_id", "firm_id", "year"],
        vcov="iid",
    ),
]

BENCHMARKERS = [
    FixestFeolsBenchmarker("fixest-map"),
    JuliaFeolsBenchmarker("FEM.jl (lsmr)"),
    PyFeolsBenchmarkerFullApi("pyfixest (rust-cg)", "rust-cg", fixef_maxiter=10000),
    PyFeolsBenchmarkerFullApi("pyfixest (rust-map)", "rust", fixef_maxiter=10000),
]


def _n_obs_for_akm_scenario(dgp_name: str) -> int:
    return SCALE_N_OBS.get(dgp_name, DEFAULT_N_OBS)


def generate_akm_datasets():
    datasets = []
    for dgp in DGPS:
        n_obs = _n_obs_for_akm_scenario(dgp.dgp_name)
        print(f"[data] generating {dgp.dgp_name} n={n_obs:,}")
        datasets.extend(dgp.generate(n=n_obs, n_iters=N_ITERS, burn_in=BURN_IN))
    print(f"[data] {len(datasets)} datasets ready")
    return datasets


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = generate_akm_datasets()
    results = run_benchmarks(BENCHMARKERS, datasets, SPECS)
    export_and_plot(results, OUTPUT_CSV, figure_dir=FIGURE_DIR)
