from pathlib import Path

import pandas as pd

from benchmarks.modular.plotting import plot_benchmarks


def test_plot_benchmarks_writes_one_figure_per_dgp(tmp_path: Path):
    results_df = pd.DataFrame(
        {
            "dgp": [
                "simple",
                "simple",
                "simple",
                "simple",
                "akm_low_mobility",
                "akm_low_mobility",
                "akm_low_mobility",
                "akm_low_mobility",
            ],
            "n_fe": [2, 2, 3, 3, 2, 2, 3, 3],
            "n_obs": [1_000, 10_000, 1_000, 10_000, 1_000, 10_000, 1_000, 10_000],
            "backend": [
                "pyfixest",
                "pyfixest",
                "pyfixest",
                "pyfixest",
                "fixest",
                "fixest",
                "fixest",
                "fixest",
            ],
            "time": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
        }
    )

    output_path = tmp_path / "feols_bench.png"
    plot_benchmarks(results_df, output_path)

    expected_paths = [
        tmp_path / "feols_bench_simple.png",
        tmp_path / "feols_bench_simple_bars.png",
        tmp_path / "feols_bench_akm_low_mobility.png",
        tmp_path / "feols_bench_akm_low_mobility_bars.png",
    ]

    assert all(path.exists() for path in expected_paths)
