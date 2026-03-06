from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .dgps import BipartiteDGP, DifficultDGP
    from .feols_benchmarkers import (
        FixestFeolsBenchmarker,
        JuliaFeolsBenchmarker,
        PyFeolsBenchmarker,
    )
    from .interfaces import FeolsBenchmarkerProtocol, FeolsSpec
    from .plotting import plot_benchmarks
except ImportError:
    from dgps import BipartiteDGP, DifficultDGP
    from feols_benchmarkers import (
        FixestFeolsBenchmarker,
        JuliaFeolsBenchmarker,
        PyFeolsBenchmarker,
    )
    from interfaces import FeolsBenchmarkerProtocol, FeolsSpec
    from plotting import plot_benchmarks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES = [1_000, 10_000, 100_000]
N_ITERS = 3
BURN_IN = 1
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "feols_bench.csv"

FEOLS_SPECS = [
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


def _make_benchmarkers() -> list[FeolsBenchmarkerProtocol]:
    from pyfixest.core.demean import demean as demean_rust

    return [
        PyFeolsBenchmarker("rust-ap", demean_rust),
        FixestFeolsBenchmarker(),
        JuliaFeolsBenchmarker(),
    ]


def _serialize_result(result) -> dict:
    """Convert FeolsResult to a flat dict suitable for CSV."""
    d = asdict(result)
    substeps = d.pop("substeps", None)
    if substeps:
        for key, val in substeps.items():
            d[f"substep_{key}"] = val
    return d


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    dgps = [
        DifficultDGP(DATA_DIR),
        BipartiteDGP(DATA_DIR, n_time=10, firm_size=5, p_move=0.05, c_sort=3.0),
    ]

    all_datasets = []
    for dgp in dgps:
        for n in SIZES:
            print(f"[data] generating {dgp.dgp_name} n={n:,}")
            all_datasets.extend(dgp.generate(n=n, n_iters=N_ITERS, burn_in=BURN_IN))

    print(f"[data] {len(all_datasets)} datasets ready")

    benchmarkers = _make_benchmarkers()
    all_results = []
    for benchmarker in benchmarkers:
        for spec in FEOLS_SPECS:
            results = benchmarker.run(all_datasets, spec)
            all_results.extend(results)

    results_df = pd.DataFrame([_serialize_result(r) for r in all_results])
    results_df = results_df[results_df["iter_type"] != "burnin"].copy()
    results_df.to_csv(OUTPUT_CSV, index=False)

    plot_df = results_df[results_df["success"] & results_df["time"].notna()].copy()
    plot_benchmarks(plot_df, OUTPUT_CSV.with_suffix(".png"))


if __name__ == "__main__":
    main()
