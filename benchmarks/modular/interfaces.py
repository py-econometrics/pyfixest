from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class BenchmarkDataset:
    """High-level dataset descriptor shared across benchmark backends."""

    dataset_id: str
    data_path: Path
    dgp: str
    n_obs: int
    iter_type: str
    iter_num: int


class DataGeneratorProtocol(Protocol):
    @property
    def dgp_name(self) -> str: ...

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        """Generate datasets for one (dgp, n) combination."""
        ...


@dataclass(frozen=True)
class FeolsSpec:
    """Specification for a full feols pipeline benchmark."""

    depvar: str
    covariates: list[str]
    fe_cols: list[str]
    vcov: str | dict[str, str]

    @property
    def formula(self) -> str:
        """Build fixest-style formula: y ~ x1 | indiv_id + year."""
        rhs = " + ".join(self.covariates) if self.covariates else "1"
        if self.fe_cols:
            return f"{self.depvar} ~ {rhs} | {' + '.join(self.fe_cols)}"
        return f"{self.depvar} ~ {rhs}"

    @property
    def n_fe(self) -> int:
        return len(self.fe_cols)


@dataclass(frozen=True)
class FeolsResult:
    """Result row for a full feols pipeline benchmark."""

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
    substeps: dict[str, float] | None = None


class FeolsBenchmarkerProtocol(Protocol):
    @property
    def name(self) -> str: ...

    def run(
        self, datasets: list[BenchmarkDataset], spec: FeolsSpec
    ) -> list[FeolsResult]:
        """Benchmark one feols backend on a list of datasets for a fixed spec."""
        ...
