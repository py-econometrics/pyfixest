from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from interfaces import BenchmarkDataset, BenchmarkSpec, DemeanResult


def _result_from_dataset(
    dataset: BenchmarkDataset,
    spec: BenchmarkSpec,
    *,
    backend: str,
    elapsed: float | None,
    success: bool,
    error: str | None = None,
) -> DemeanResult:
    return DemeanResult(
        dataset_id=dataset.dataset_id,
        iter_type=dataset.iter_type,
        iter_num=dataset.iter_num,
        dgp=dataset.dgp,
        n_obs=dataset.n_obs,
        n_fe=len(spec.fe_cols),
        backend=backend,
        time=elapsed,
        success=success,
        error=error,
    )


class PyFixestDemeaner:
    def __init__(self, backend: str = "numba"):
        self.backend = backend

    @property
    def name(self) -> str:
        return f"pyfixest.{self.backend}"

    def run(
        self, datasets: list[BenchmarkDataset], spec: BenchmarkSpec
    ) -> list[DemeanResult]:
        from pyfixest.estimation.internals.demean_ import _set_demeaner_backend

        demean_func = _set_demeaner_backend(self.backend)
        results: list[DemeanResult] = []
        cols = [*spec.demean_cols, *spec.fe_cols]

        for dataset in datasets:
            try:
                df = pd.read_parquet(dataset.data_path, columns=cols)
                x = df[spec.demean_cols].to_numpy(dtype=np.float64, copy=False)
                flist = df[spec.fe_cols].to_numpy(dtype=np.uint64, copy=False)
                weights = np.ones(x.shape[0], dtype=np.float64)

                start = time.perf_counter()
                _, success = demean_func(x, flist, weights)
                elapsed = time.perf_counter() - start

                results.append(
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=elapsed,
                        success=bool(success),
                    )
                )
            except Exception as exc:
                results.append(
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                    )
                )

        return results


def _parse_subprocess_output(
    *,
    datasets: list[BenchmarkDataset],
    spec: BenchmarkSpec,
    backend: str,
    completed_process: subprocess.CompletedProcess[str],
) -> list[DemeanResult]:
    parsed_by_id: dict[str, dict] = {}

    for line in completed_process.stdout.splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            entry = json.loads(payload)
        except json.JSONDecodeError:
            continue
        dataset_id = entry.get("dataset_id")
        if isinstance(dataset_id, str):
            parsed_by_id[dataset_id] = entry

    default_error = completed_process.stderr.strip() or None
    results: list[DemeanResult] = []

    for dataset in datasets:
        entry = parsed_by_id.get(dataset.dataset_id)
        if entry is None:
            missing_error = default_error or "No result emitted by subprocess backend."
            results.append(
                _result_from_dataset(
                    dataset,
                    spec,
                    backend=backend,
                    elapsed=None,
                    success=False,
                    error=missing_error,
                )
            )
            continue

        elapsed_raw = entry.get("time")
        try:
            elapsed = None if elapsed_raw is None else float(elapsed_raw)
        except (TypeError, ValueError):
            elapsed = None

        results.append(
            _result_from_dataset(
                dataset,
                spec,
                backend=backend,
                elapsed=elapsed,
                success=bool(entry.get("success", elapsed is not None)),
                error=entry.get("error"),
            )
        )

    return results


class SubprocessDemeaner:
    """Generic subprocess backend (R/Julia/etc.) using JSON config + JSONL stdout."""

    def __init__(
        self,
        *,
        name: str,
        command_prefix: Sequence[str],
        script_path: Path,
    ):
        self._name = name
        self._command_prefix = tuple(command_prefix)
        self._script_path = script_path.resolve()

    @property
    def name(self) -> str:
        return self._name

    def run(
        self, datasets: list[BenchmarkDataset], spec: BenchmarkSpec
    ) -> list[DemeanResult]:
        manifest = [
            {
                "dataset_id": dataset.dataset_id,
                "data_path": str(dataset.data_path.resolve()),
                "dgp": dataset.dgp,
                "n_obs": dataset.n_obs,
                "iter_type": dataset.iter_type,
                "iter_num": dataset.iter_num,
            }
            for dataset in datasets
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest": manifest,
                        "demean_cols": spec.demean_cols,
                        "fe_cols": spec.fe_cols,
                    }
                ),
                encoding="utf-8",
            )

            command = [
                *self._command_prefix,
                str(self._script_path),
                str(config_path),
            ]

            try:
                proc = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:
                return [
                    _result_from_dataset(
                        dataset,
                        spec,
                        backend=self.name,
                        elapsed=None,
                        success=False,
                        error=str(exc),
                    )
                    for dataset in datasets
                ]

        return _parse_subprocess_output(
            datasets=datasets,
            spec=spec,
            backend=self.name,
            completed_process=proc,
        )


class FixestDemeaner:
    def __init__(self, script_path: Path | None = None):
        resolved_script = script_path or Path(__file__).with_name("demean_r.R")
        self._delegate = SubprocessDemeaner(
            name="r.fixest",
            command_prefix=["Rscript"],
            script_path=resolved_script,
        )

    @property
    def name(self) -> str:
        return self._delegate.name

    def run(
        self, datasets: list[BenchmarkDataset], spec: BenchmarkSpec
    ) -> list[DemeanResult]:
        return self._delegate.run(datasets, spec)
