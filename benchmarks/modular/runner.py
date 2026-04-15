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


def generate_datasets(
    dgps: list[DataGeneratorProtocol],
    sizes: list[int],
    n_iters: int,
    burn_in: int,
) -> list[BenchmarkDataset]:
    all_datasets: list[BenchmarkDataset] = []
    for dgp in dgps:
        for n in sizes:
            all_datasets.extend(dgp.generate(n=n, n_iters=n_iters, burn_in=burn_in))
    print(f"[data] {len(all_datasets)} datasets ready")
    return all_datasets


def run_benchmarks(
    benchmarkers: list[FeolsBenchmarkerProtocol],
    datasets: list[BenchmarkDataset],
    specs: list[FeolsSpec],
    output_csv: Path,
) -> pd.DataFrame:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for benchmarker in benchmarkers:
        csv_path = _backend_output_csv(output_csv, benchmarker.name)
        if csv_path.exists():
            print(f"[skip] {benchmarker.name}: {csv_path.name} already exists")
            frames.append(pd.read_csv(csv_path))
            continue
        backend_results: list[FeolsResult] = []
        for spec in specs:
            spec_datasets = [dataset for dataset in datasets if dataset.k >= spec.k]
            backend_results.extend(benchmarker.run(spec_datasets, spec))
        if not backend_results:
            continue
        df = pd.DataFrame([_serialize_result(r) for r in backend_results])
        df = df[df["iter_type"] != "burnin"].copy()
        df.to_csv(csv_path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_results(
    results_df: pd.DataFrame,
    output_csv: Path,
    *,
    figure_dir: Path | None = None,
    figure_backends: list[str] | None = None,
) -> None:
    plot_df = results_df[results_df["success"] & results_df["time"].notna()].copy()
    plot_benchmarks(
        plot_df,
        output_csv.with_suffix(".png"),
        figure_dir=figure_dir,
        figure_backends=figure_backends,
    )
