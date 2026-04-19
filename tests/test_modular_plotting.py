from pathlib import Path

import pandas as pd

import benchmarks.modular.plotting as plotting
from benchmarks.modular.plotting import plot_benchmarks


def test_build_styles_assigns_distinct_style_to_full_backend_set() -> None:
    backends = [
        "FEM.jl (lsmr)",
        "fixest-map",
        "pyfixest (cupy32)",
        "pyfixest (jax)",
        "pyfixest (rust-cg)",
        "pyfixest (rust-map)",
        "pyfixest (scipy-lsmr)",
        "pyfixest (torch-cpu)",
        "pyfixest (torch-cuda)",
    ]

    styles = plotting._build_styles(backends)
    style_pairs = [
        (style["color"], style["marker"]) for style in styles.values()
    ]

    assert len(style_pairs) == len(set(style_pairs))
    assert (
        styles["FEM.jl (lsmr)"]["color"],
        styles["FEM.jl (lsmr)"]["marker"],
    ) != (
        styles["pyfixest (torch-cuda)"]["color"],
        styles["pyfixest (torch-cuda)"]["marker"],
    )


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
            "model_k": [1] * 8,
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
                "akm_interaction_1",
                "akm_interaction_1",
                "akm_freeze_1",
                "akm_freeze_1",
            ],
            "model_k": [1] * 12,
            "n_fe": [2, 3] * 6,
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
            ],
            "backend": ["fixest"] * 12,
            "time": [
                1.0,
                1.5,
                0.4,
                0.6,
                1.2,
                1.8,
                0.9,
                1.3,
                1.3,
                1.9,
                1.1,
                1.7,
            ],
        }
    )

    output_path = tmp_path / "feols_akm_sweep.png"
    plot_benchmarks(results_df, output_path)

    expected_paths = [
        tmp_path / "feols_akm_sweep_akm_sweep_scale.png",
        tmp_path / "feols_akm_sweep_akm_sweep_sorting.png",
        tmp_path / "feols_akm_sweep_akm_sweep_mobility.png",
        tmp_path / "feols_akm_sweep_akm_sweep_interaction.png",
        tmp_path / "feols_akm_sweep_akm_sweep_freeze.png",
    ]

    assert all(path.exists() for path in expected_paths)


def test_plot_benchmarks_writes_occupation_family_figures(tmp_path: Path):
    results_df = pd.DataFrame(
        {
            "dgp": [
                "akm_baseline",
                "akm_occlambda_2",
                "akm_occsize_3",
            ],
            "model_k": [1, 1, 1],
            "n_fe": [4, 4, 4],
            "n_obs": [1_000_000] * 3,
            "backend": ["fixest"] * 3,
            "time": [1.3, 1.8, 1.6],
        }
    )

    output_path = tmp_path / "feols_akm_occupation.png"
    plot_benchmarks(results_df, output_path)

    expected_paths = [
        tmp_path / "feols_akm_occupation_akm_sweep_occlambda.png",
        tmp_path / "feols_akm_occupation_akm_sweep_occsize.png",
    ]

    assert all(path.exists() for path in expected_paths)


def test_plot_benchmarks_filters_to_requested_backends(
    tmp_path: Path, monkeypatch
) -> None:
    results_df = pd.DataFrame(
        {
            "dgp": ["simple", "simple", "simple", "simple"],
            "model_k": [1, 1, 1, 1],
            "n_fe": [2, 2, 3, 3],
            "n_obs": [1_000, 10_000, 1_000, 10_000],
            "backend": ["pyfixest", "fixest", "pyfixest", "fixest"],
            "time": [0.1, 0.2, 0.3, 0.4],
        }
    )
    captured_backends: list[list[str]] = []

    def fake_plot_dgp_figure(*args, **kwargs) -> None:
        dgp_summary = args[0]
        captured_backends.append(sorted(dgp_summary["backend"].unique().tolist()))

    monkeypatch.setattr(plotting, "_plot_dgp_figure", fake_plot_dgp_figure)

    plotting.plot_benchmarks(
        results_df,
        tmp_path / "feols_bench.png",
        figure_backends=["pyfixest"],
    )

    assert captured_backends == [["pyfixest"]]


def test_plot_benchmarks_passes_multiple_k_values_to_generic_figures(
    tmp_path: Path, monkeypatch
) -> None:
    results_df = pd.DataFrame(
        {
            "dgp": ["simple"] * 8,
            "model_k": [1, 1, 1, 1, 5, 5, 5, 5],
            "n_fe": [2, 2, 3, 3, 2, 2, 3, 3],
            "n_obs": [1_000, 10_000, 1_000, 10_000] * 2,
            "backend": ["pyfixest"] * 8,
            "time": [0.1, 0.2, 0.3, 0.4, 0.11, 0.21, 0.31, 0.41],
        }
    )
    captured_k_values: list[list[int]] = []

    def fake_plot_dgp_figure(*args, **kwargs) -> None:
        dgp_summary = args[0]
        captured_k_values.append(sorted(dgp_summary["model_k"].unique().tolist()))

    monkeypatch.setattr(plotting, "_plot_dgp_figure", fake_plot_dgp_figure)

    plotting.plot_benchmarks(results_df, tmp_path / "feols_bench.png")

    assert captured_k_values == [[1, 5]]


def test_plot_benchmarks_defaults_missing_model_k_to_one(
    tmp_path: Path, monkeypatch
) -> None:
    results_df = pd.DataFrame(
        {
            "dgp": ["simple", "simple"],
            "model_k": [None, None],
            "n_fe": [2, 3],
            "n_obs": [1_000, 10_000],
            "backend": ["pyfixest", "pyfixest"],
            "time": [0.1, 0.3],
        }
    )
    captured_k_values: list[list[int]] = []

    def fake_plot_dgp_figure(*args, **kwargs) -> None:
        dgp_summary = args[0]
        captured_k_values.append(sorted(dgp_summary["model_k"].unique().tolist()))

    monkeypatch.setattr(plotting, "_plot_dgp_figure", fake_plot_dgp_figure)

    plotting.plot_benchmarks(results_df, tmp_path / "feols_bench.png")

    assert captured_k_values == [[1]]
