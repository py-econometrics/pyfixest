from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from interfaces import (
    BenchmarkDataset,
    DataGeneratorProtocol,
    FeolsBenchmarkerProtocol,
    FeolsResult,
    FeolsSpec,
)
from plotting import plot_benchmarks


def _serialize_result(result: FeolsResult) -> dict:
    d = asdict(result)
    substeps = d.pop("substeps", None)
    if substeps:
        for key, val in substeps.items():
            d[f"substep_{key}"] = val
    return d


def generate_datasets(
    dgps: list[DataGeneratorProtocol],
    sizes: list[int],
    n_iters: int,
    burn_in: int,
) -> list[BenchmarkDataset]:
    all_datasets: list[BenchmarkDataset] = []
    for dgp in dgps:
        for n in sizes:
            print(f"[data] generating {dgp.dgp_name} n={n:,}")
            all_datasets.extend(dgp.generate(n=n, n_iters=n_iters, burn_in=burn_in))
    print(f"[data] {len(all_datasets)} datasets ready")
    return all_datasets


def run_benchmarks(
    benchmarkers: list[FeolsBenchmarkerProtocol],
    datasets: list[BenchmarkDataset],
    specs: list[FeolsSpec],
) -> list[FeolsResult]:
    all_results: list[FeolsResult] = []
    for benchmarker in benchmarkers:
        for spec in specs:
            results = benchmarker.run(datasets, spec)
            all_results.extend(results)
    return all_results


def export_and_plot(
    results: list[FeolsResult],
    output_csv: Path,
    *,
    figure_dir: Path | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([_serialize_result(r) for r in results])
    results_df = results_df[results_df["iter_type"] != "burnin"].copy()
    results_df.to_csv(output_csv, index=False)

    plot_df = results_df[results_df["success"] & results_df["time"].notna()].copy()
    plot_benchmarks(plot_df, output_csv.with_suffix(".png"), figure_dir=figure_dir)
