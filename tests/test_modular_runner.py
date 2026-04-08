from pathlib import Path

import benchmarks.modular.runner as runner
from benchmarks.modular.interfaces import FeolsResult


def test_export_and_plot_writes_one_csv_per_backend_and_combines_for_plotting(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_plot_benchmarks(results_df, output_path, **kwargs) -> None:
        captured["results_df"] = results_df.copy()
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(runner, "plot_benchmarks", fake_plot_benchmarks)

    results = [
        FeolsResult(
            dataset_id="simple_1000_1",
            iter_type="measure",
            iter_num=1,
            dgp="simple",
            n_obs=1_000,
            n_fe=2,
            backend="pyfixest (rust-cg)",
            time=0.1,
            success=True,
        ),
        FeolsResult(
            dataset_id="simple_1000_1",
            iter_type="measure",
            iter_num=1,
            dgp="simple",
            n_obs=1_000,
            n_fe=2,
            backend="fixest-map",
            time=0.2,
            success=True,
        ),
        FeolsResult(
            dataset_id="simple_1000_0",
            iter_type="burnin",
            iter_num=0,
            dgp="simple",
            n_obs=1_000,
            n_fe=2,
            backend="fixest-map",
            time=9.9,
            success=True,
        ),
    ]

    output_csv = tmp_path / "feols_akm_sweep.csv"
    runner.export_and_plot(results, output_csv)

    expected_csvs = [
        tmp_path / "feols_akm_sweep__fixest_map.csv",
        tmp_path / "feols_akm_sweep__pyfixest_rust_cg.csv",
    ]
    assert all(path.exists() for path in expected_csvs)

    combined_df = captured["results_df"]
    assert sorted(combined_df["backend"].unique().tolist()) == [
        "fixest-map",
        "pyfixest (rust-cg)",
    ]
    assert set(combined_df["iter_type"]) == {"measure"}
    assert captured["output_path"] == tmp_path / "feols_akm_sweep.png"
