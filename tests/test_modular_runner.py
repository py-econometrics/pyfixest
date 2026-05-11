from pathlib import Path

import benchmarks.modular.runner as runner
from benchmarks.modular.interfaces import BenchmarkDataset, FeolsResult, FeolsSpec


def test_run_benchmarks_writes_per_backend_csv_and_skips_existing(
    tmp_path: Path,
) -> None:
    run_count = {"a": 0, "b": 0}

    class BenchmarkerA:
        @property
        def name(self) -> str:
            return "backend-a"

        def run(self, datasets, spec):
            run_count["a"] += 1
            return [
                FeolsResult(
                    source_dataset_id="d1",
                    source_k=1,
                    iter_type="measure",
                    iter_num=1,
                    dgp="simple",
                    model_k=1,
                    n_obs=1_000,
                    n_fe=2,
                    backend="backend-a",
                    time=0.1,
                    success=True,
                ),
            ]

    class BenchmarkerB:
        @property
        def name(self) -> str:
            return "backend-b"

        def run(self, datasets, spec):
            run_count["b"] += 1
            return [
                FeolsResult(
                    source_dataset_id="d1",
                    source_k=1,
                    iter_type="measure",
                    iter_num=1,
                    dgp="simple",
                    model_k=1,
                    n_obs=1_000,
                    n_fe=2,
                    backend="backend-b",
                    time=0.2,
                    success=True,
                ),
            ]

    datasets = [
        BenchmarkDataset(
            dataset_id="d1",
            data_path=tmp_path / "d1.parquet",
            dgp="simple",
            k=1,
            n_obs=1_000,
            iter_type="iter",
            iter_num=1,
        ),
    ]
    specs = [
        FeolsSpec(
            depvar="y",
            covariates=["x1"],
            fe_cols=["indiv_id", "year"],
            vcov="iid",
        ),
    ]
    output_csv = tmp_path / "bench.csv"

    # First run: both backends execute
    df = runner.run_benchmarks(
        [BenchmarkerA(), BenchmarkerB()], datasets, specs, output_csv
    )
    assert run_count == {"a": 1, "b": 1}
    assert sorted(df["backend"].unique()) == ["backend-a", "backend-b"]
    assert (tmp_path / "bench__backend_a.csv").exists()
    assert (tmp_path / "bench__backend_b.csv").exists()

    # Second run: both backends are skipped (CSVs exist)
    df2 = runner.run_benchmarks(
        [BenchmarkerA(), BenchmarkerB()], datasets, specs, output_csv
    )
    assert run_count == {"a": 1, "b": 1}  # no additional runs
    assert sorted(df2["backend"].unique()) == ["backend-a", "backend-b"]


def test_run_benchmarks_matches_datasets_to_specs_by_supported_k(
    tmp_path: Path,
) -> None:
    captured = []

    class DummyBenchmarker:
        @property
        def name(self) -> str:
            return "dummy"

        def run(self, datasets, spec):
            captured.append((spec.k, [dataset.k for dataset in datasets]))
            return []

    datasets = [
        BenchmarkDataset(
            dataset_id="simple_1000_k10_iter_1",
            data_path=tmp_path / "simple_k10.parquet",
            dgp="simple",
            k=10,
            n_obs=1_000,
            iter_type="iter",
            iter_num=1,
        ),
        BenchmarkDataset(
            dataset_id="simple_1000_k5_iter_1",
            data_path=tmp_path / "simple_k5.parquet",
            dgp="simple",
            k=5,
            n_obs=1_000,
            iter_type="iter",
            iter_num=1,
        ),
    ]
    specs = [
        FeolsSpec(
            depvar="y",
            covariates=["x1"],
            fe_cols=["indiv_id", "year"],
            vcov="iid",
        ),
        FeolsSpec(
            depvar="y",
            covariates=["x1", "x2", "x3", "x4", "x5"],
            fe_cols=["indiv_id", "year"],
            vcov="iid",
        ),
    ]

    output_csv = tmp_path / "bench.csv"
    runner.run_benchmarks([DummyBenchmarker()], datasets, specs, output_csv)

    assert captured == [(1, [10, 5]), (5, [10, 5])]
