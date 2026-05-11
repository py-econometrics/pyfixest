from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MODULAR_DIR = Path(__file__).resolve().parents[1] / "modular"
if str(MODULAR_DIR) not in sys.path:
    sys.path.insert(0, str(MODULAR_DIR))

import plotting  # noqa: E402
import runner  # noqa: E402


def test_run_benchmarks_reuses_current_schema_csv(tmp_path: Path) -> None:
    """Test that cached CSV reuse works directly with the current result schema."""

    class DummyBenchmarker:
        name = "fixest-map"

        def run(self, datasets, spec):
            raise AssertionError("cached CSV should have prevented benchmark execution")

    output_csv = tmp_path / "feols_bench.csv"
    cached_csv = tmp_path / "feols_bench__fixest_map.csv"
    current = pd.DataFrame(
        [
            {
                "source_dataset_id": "simple_1000_k10_iter_1",
                "source_k": 10,
                "iter_type": "iter",
                "iter_num": 1,
                "dgp": "simple",
                "model_k": 5,
                "n_obs": 1000,
                "n_fe": 2,
                "backend": "fixest-map",
                "time": 0.1,
                "success": True,
                "error": None,
            }
        ]
    )
    current.to_csv(cached_csv, index=False)

    results = runner.run_benchmarks([DummyBenchmarker()], [], [], output_csv)

    assert "source_dataset_id" in results.columns
    assert "model_k" in results.columns
    assert "source_k" in results.columns
    assert results.loc[0, "source_k"] == 10
    assert results.loc[0, "model_k"] == 5


def test_plot_benchmarks_accepts_model_k_schema(tmp_path: Path) -> None:
    """Test that plotting consumes the current benchmark result schema."""
    results_df = pd.DataFrame(
        [
            {
                "source_dataset_id": "simple_1000_k10_iter_1",
                "source_k": 10,
                "iter_type": "iter",
                "iter_num": 1,
                "dgp": "simple",
                "model_k": 1,
                "n_obs": 1000,
                "n_fe": 2,
                "backend": "fixest-map",
                "time": 0.027,
                "success": True,
                "error": None,
            },
            {
                "source_dataset_id": "simple_10000_k10_iter_1",
                "source_k": 10,
                "iter_type": "iter",
                "iter_num": 1,
                "dgp": "simple",
                "model_k": 1,
                "n_obs": 10000,
                "n_fe": 2,
                "backend": "fixest-map",
                "time": 0.030,
                "success": True,
                "error": None,
            },
        ]
    )

    plotting.plot_benchmarks(results_df, tmp_path / "bench.png")

    assert (tmp_path / "bench_simple.png").exists()
