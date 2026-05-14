from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from feols_benchmarkers import (
    FixestFeolsBenchmarker,
    JuliaFeolsBenchmarker,
    PyFeolsBenchmarkerFullApi,
)
from interfaces import BenchmarkDataset, FeolsSpec
from runner import run_benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "benchmarks" / "correia-benchmark-data"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUT_CSV = PROJECT_ROOT / "benchmarks" / "results" / "correia-benchmarks.csv"
SCRIPT_DIR = Path(__file__).resolve().parent
N_ITERATIONS = 3
TOLERANCE = 1e-8
R_FIXEF_ITERATIONS = 100_000

DATASETS = [
    "credit2",
    "credit",
    "soccer",
    "synthetic-complete",
    "synthetic-uniform-easy",
    "synthetic-uniform-hard",
    "synthetic-uniform-harder",
    "synthetic-assortative",
    "synthetic-zigzag",
    "enron",
    "github",
    "patents",
    "workers",
    "schools",
    "directors",
]
SPEC = FeolsSpec(
    depvar="y",
    covariates=["x1", "x2"],
    fe_cols=["id1", "id2"],
    vcov="iid",
)
LANGUAGE = {
    "pyfixest-map": "Python",
    "pyfixest-within": "Python",
    "fixest": "R",
    "FixedEffectModels": "Julia",
}


def fmt_time(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.1f}ms"
    return f"{value:.3f}s"


def _n_obs_from_metadata(dataset: str) -> int:
    metadata_path = METADATA_DIR / f"{dataset}.json"
    with metadata_path.open() as f:
        return int(json.load(f)["rows"])


def load_datasets() -> list[BenchmarkDataset]:
    datasets = []
    for dataset in DATASETS:
        data_path = DATA_DIR / f"{dataset}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing Correia CSV: {data_path}")
        n_obs = _n_obs_from_metadata(dataset)
        for iter_num in range(1, N_ITERATIONS + 1):
            datasets.append(
                BenchmarkDataset(
                    dataset_id=dataset,
                    data_path=data_path.resolve(),
                    dgp=dataset,
                    k=len(SPEC.covariates),
                    n_obs=n_obs,
                    iter_type="iter",
                    iter_num=iter_num,
                )
            )
    return datasets


def build_benchmarkers() -> list:
    import pyfixest as pf

    extra_config = {"tolerance": TOLERANCE, "fixef_iterations": R_FIXEF_ITERATIONS}
    return [
        PyFeolsBenchmarkerFullApi(
            "pyfixest-map",
            demeaner=pf.MapDemeaner(fixef_tol=TOLERANCE, fixef_maxiter=100_000),
        ),
        PyFeolsBenchmarkerFullApi(
            "pyfixest-within", demeaner=pf.WithinDemeaner(fixef_tol=TOLERANCE)
        ),
        FixestFeolsBenchmarker(
            "fixest",
            script_path=SCRIPT_DIR / "correia_r.R",
            extra_config=extra_config,
        ),
        JuliaFeolsBenchmarker(
            "FixedEffectModels",
            script_path=SCRIPT_DIR / "correia_julia.jl",
            extra_config=extra_config,
        ),
    ]


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in results_df.groupby(
        ["source_dataset_id", "backend", "n_obs", "n_fe"], sort=False
    ):
        dataset, backend, n_obs, n_fe = keys
        successful = group[group["success"] & group["time"].notna()]
        success = not successful.empty
        errors = group["error"].dropna()
        rows.append(
            {
                "source_dataset_id": dataset,
                "backend": backend,
                "n_obs": n_obs,
                "n_fe": n_fe,
                "time": successful["time"].median() if success else pd.NA,
                "success": success,
                "error": pd.NA if success or errors.empty else errors.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def format_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": results_df["source_dataset_id"],
            "language": results_df["backend"].map(LANGUAGE),
            "algo": results_df["backend"],
            "n_obs": results_df["n_obs"],
            "n_fe": results_df["n_fe"],
            "model": SPEC.formula,
            "time": results_df["time"],
            "success": results_df["success"],
            "error": results_df["error"],
        }
    )


def print_runtime_summary(results_df: pd.DataFrame) -> None:
    dataset_width = max(12, max(len(dataset) for dataset in DATASETS))
    header = (
        f"{'dataset':<{dataset_width}} {'language':<7} {'algo':<17} "
        f"{'n_obs':>10} {'n_fe':>4} {'time':>10}  status"
    )
    separator = "-" * len(header)

    print("\n  Correia runtimes", flush=True)
    print(f"  {separator}", flush=True)
    print(f"  {header}", flush=True)
    print(f"  {separator}", flush=True)

    backend_order = {name: idx for idx, name in enumerate(LANGUAGE)}
    dataset_order = {name: idx for idx, name in enumerate(DATASETS)}
    summary_df = results_df.sort_values(
        by=["backend", "source_dataset_id"],
        key=lambda col: col.map(
            backend_order if col.name == "backend" else dataset_order
        ).fillna(len(DATASETS)),
    )

    for _, row in summary_df.iterrows():
        ok = bool(row["success"])
        elapsed = row["time"]
        time_text = fmt_time(elapsed) if ok and pd.notna(elapsed) else "—"
        status = "ok" if ok else str(row["error"])[:40]
        print(
            "  "
            f"{row['source_dataset_id']:<{dataset_width}} "
            f"{LANGUAGE[row['backend']]:<7} "
            f"{row['backend']:<17} "
            f"{int(row['n_obs']):>10,} "
            f"{int(row['n_fe']):>4} "
            f"{time_text:>10}  "
            f"{status}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[correia] using tolerance {TOLERANCE:g}; convergence criteria differ by backend",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_results = run_benchmarks(
            build_benchmarkers(),
            load_datasets(),
            [SPEC],
            Path(tmpdir) / "correia-benchmarks-raw.csv",
        )
    summary_results = summarize_results(raw_results)
    print_runtime_summary(summary_results)
    format_results(summary_results).to_csv(OUTPUT_CSV, index=False)
    print(f"[correia] wrote {OUTPUT_CSV}", flush=True)
