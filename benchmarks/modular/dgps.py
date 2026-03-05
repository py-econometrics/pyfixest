from __future__ import annotations

from pathlib import Path

from dgp_functions import base_dgp
from interfaces import BenchmarkDataset


def _seed_for(dgp_name: str, n: int, iteration: int) -> int:
    """Build deterministic seeds so benchmark runs are reproducible."""
    dgp_offset = 0 if dgp_name == "simple" else 1
    return n * 100 + iteration * 17 + dgp_offset


def _generate_datasets(
    *,
    dgp_name: str,
    dgp_type: str,
    n: int,
    n_iters: int,
    burn_in: int,
    data_dir: Path,
) -> list[BenchmarkDataset]:
    """Generate parquet-backed datasets for one DGP/size combination."""
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[BenchmarkDataset] = []

    for i in range(1, burn_in + n_iters + 1):
        iter_type = "burnin" if i <= burn_in else "iter"
        iter_num = i if i <= burn_in else i - burn_in
        dataset_id = f"{dgp_name}_{n}_{iter_type}_{iter_num}"
        data_path = data_dir / f"{dataset_id}.parquet"

        df = base_dgp(n=n, type_=dgp_type, seed=_seed_for(dgp_name, n, i))
        df.to_parquet(data_path)

        datasets.append(
            BenchmarkDataset(
                dataset_id=dataset_id,
                data_path=data_path.resolve(),
                dgp=dgp_name,
                n_obs=len(df),
                iter_type=iter_type,
                iter_num=iter_num,
            )
        )

    return datasets


class SimpleDGP:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    @property
    def dgp_name(self) -> str:
        return "simple"

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        return _generate_datasets(
            dgp_name=self.dgp_name,
            dgp_type="simple",
            n=n,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
        )


class DifficultDGP:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    @property
    def dgp_name(self) -> str:
        return "difficult"

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        return _generate_datasets(
            dgp_name=self.dgp_name,
            dgp_type="difficult",
            n=n,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
        )
