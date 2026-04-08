from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

try:
    from .interfaces import (
        BenchmarkDataset,
        DataGeneratorProtocol,
        FeolsBenchmarkerProtocol,
        FeolsResult,
        FeolsSpec,
    )
    from .plotting import plot_benchmarks
except ImportError:
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


def _backend_slug(backend: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", backend).strip("_").lower()
    return slug or "backend"


def _backend_output_csv(output_csv: Path, backend: str) -> Path:
    return output_csv.with_name(f"{output_csv.stem}__{_backend_slug(backend)}.csv")


def _matching_backend_csvs(output_csv: Path) -> list[Path]:
    pattern = f"{output_csv.stem}__*.csv"
    return sorted(output_csv.parent.glob(pattern))


def _load_combined_results(output_csv: Path) -> pd.DataFrame:
    csv_paths = _matching_backend_csvs(output_csv)
    if not csv_paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in csv_paths), ignore_index=True)


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
    figure_backends: list[str] | None = None,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame([_serialize_result(r) for r in results])
    results_df = results_df[results_df["iter_type"] != "burnin"].copy()
    for backend, backend_df in results_df.groupby("backend", sort=True):
        backend_df.to_csv(_backend_output_csv(output_csv, backend), index=False)

    combined_results_df = _load_combined_results(output_csv)
    plot_df = combined_results_df[
        combined_results_df["success"] & combined_results_df["time"].notna()
    ].copy()
    plot_benchmarks(
        plot_df,
        output_csv.with_suffix(".png"),
        figure_dir=figure_dir,
        figure_backends=figure_backends,
    )
