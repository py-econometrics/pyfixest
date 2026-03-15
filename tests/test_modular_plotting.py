from pathlib import Path

import pandas as pd

from benchmarks.modular.plotting import plot_benchmarks


def test_plot_benchmarks_writes_one_figure_per_generic_dgp(tmp_path: Path):
    results_df = pd.DataFrame(
        {
            "dgp": [
                "simple",
                "simple",
                "simple",
                "simple",
                "difficult",
                "difficult",
                "difficult",
                "difficult",
            ],
            "n_fe": [2, 2, 3, 3, 2, 2, 3, 3],
            "n_obs": [1_000, 10_000, 1_000, 10_000, 1_000, 10_000, 1_000, 10_000],
            "backend": ["pyfixest"] * 8,
            "time": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41],
        }
    )

    output_path = tmp_path / "feols_bench.png"
    plot_benchmarks(results_df, output_path)

    expected_paths = [
        tmp_path / "feols_bench_simple.png",
        tmp_path / "feols_bench_difficult.png",
    ]

    assert all(path.exists() for path in expected_paths)


def test_plot_benchmarks_writes_akm_sweep_family_figures(tmp_path: Path):
    results_df = pd.DataFrame(
        {
            "dgp": [
                "akm_baseline",
                "akm_baseline",
                "akm_scale_1",
                "akm_scale_1",
                "akm_sorting_1",
                "akm_sorting_1",
                "akm_mobility_1",
                "akm_mobility_1",
                "akm_size_1",
                "akm_size_1",
                "akm_fragmentation_2",
                "akm_fragmentation_2",
                "akm_varratio_1",
                "akm_varratio_1",
                "akm_saturation_1",
                "akm_saturation_1",
                "akm_unbalanced_1",
                "akm_unbalanced_1",
                "akm_interaction_1",
                "akm_interaction_1",
            ],
            "n_fe": [2, 3] * 10,
            "n_obs": [
                1_000_000,
                1_000_000,
                10_000,
                10_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
                1_000_000,
            ],
            "backend": ["fixest"] * 20,
            "time": [
                1.0,
                1.5,
                0.4,
                0.6,
                1.2,
                1.8,
                0.9,
                1.3,
                1.1,
                1.7,
                1.4,
                2.0,
                1.25,
                1.85,
                1.6,
                2.1,
                1.45,
                2.05,
                1.3,
                1.9,
            ],
        }
    )

    output_path = tmp_path / "feols_akm_sweep.png"
    plot_benchmarks(results_df, output_path)

    expected_paths = [
        tmp_path / "feols_akm_sweep_akm_sweep_scale.png",
        tmp_path / "feols_akm_sweep_akm_sweep_sorting.png",
        tmp_path / "feols_akm_sweep_akm_sweep_mobility.png",
        tmp_path / "feols_akm_sweep_akm_sweep_size.png",
        tmp_path / "feols_akm_sweep_akm_sweep_fragmentation.png",
        tmp_path / "feols_akm_sweep_akm_sweep_varratio.png",
        tmp_path / "feols_akm_sweep_akm_sweep_saturation.png",
        tmp_path / "feols_akm_sweep_akm_sweep_unbalanced.png",
        tmp_path / "feols_akm_sweep_akm_sweep_interaction.png",
    ]

    assert all(path.exists() for path in expected_paths)
