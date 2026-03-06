from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

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


def _generate_all_datasets(dgps, sizes, n_iters, burn_in):
    """Generate datasets for all DGP/size combinations."""
    all_datasets = []
    for dgp in dgps:
        for n in sizes:
            print(f"[data] generating {dgp.dgp_name} n={n:,}")
            all_datasets.extend(dgp.generate(n=n, n_iters=n_iters, burn_in=burn_in))
    print(f"[data] {len(all_datasets)} datasets ready")
    return all_datasets


def _run_all_benchmarks(benchmarkers, datasets, specs):
    """Run all benchmarkers across all datasets and specs."""
    all_results = []
    for benchmarker in benchmarkers:
        for spec in specs:
            results = benchmarker.run(datasets, spec)
            all_results.extend(results)
    return all_results


def _export_and_plot(results, output_csv):
    """Export results to CSV and generate plots."""
    results_df = pd.DataFrame([_serialize_result(r) for r in results])
    results_df = results_df[results_df["iter_type"] != "burnin"].copy()
    results_df.to_csv(output_csv, index=False)

    plot_df = results_df[results_df["success"] & results_df["time"].notna()].copy()
    plot_benchmarks(plot_df, output_csv.with_suffix(".png"))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    dgps = [
        DifficultDGP(DATA_DIR),
        BipartiteDGP(DATA_DIR, n_time=10, firm_size=5, p_move=0.05, c_sort=3.0),
    ]

    datasets = _generate_all_datasets(dgps, SIZES, N_ITERS, BURN_IN)
    benchmarkers = _make_benchmarkers()
    results = _run_all_benchmarks(benchmarkers, datasets, FEOLS_SPECS)
    _export_and_plot(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
