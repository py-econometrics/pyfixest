from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkDataset:
    """High-level dataset descriptor shared across all demeaning backends."""

    dataset_id: str
    data_path: Path
    dgp: str
    n_obs: int
    iter_type: str
    iter_num: int


@dataclass(frozen=True)
class BenchmarkSpec:
    """Specification for a single demeaning benchmark run."""

    demean_cols: list[str]
    fe_cols: list[str]


@dataclass(frozen=True)
class DemeanResult:
    """Result row emitted by each backend for one dataset/spec pair."""

    dataset_id: str
    iter_type: str
    iter_num: int
    dgp: str
    n_obs: int
    n_fe: int
    backend: str
    time: float | None
    success: bool
    error: str | None = None


class DataGeneratorProtocol(Protocol):
    @property
    def dgp_name(self) -> str:
        ...

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        """Generate datasets for one (dgp, n) combination."""
        ...


class DemeanerProtocol(Protocol):
    @property
    def name(self) -> str:
        ...

    def run(
        self, datasets: list[BenchmarkDataset], spec: BenchmarkSpec
    ) -> list[DemeanResult]:
        """Benchmark one backend on a list of datasets for a fixed spec."""
        ...
