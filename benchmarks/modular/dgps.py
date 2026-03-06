from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from .dgp_functions import base_dgp, simulate_bipartite
    from .interfaces import BenchmarkDataset
except ImportError:
    from dgp_functions import base_dgp, simulate_bipartite
    from interfaces import BenchmarkDataset


def _seed_for(dgp_name: str, n: int, iteration: int) -> int:
    """Build deterministic seeds so benchmark runs are reproducible."""
    dgp_offset = {"simple": 0, "difficult": 1, "bipartite": 2}.get(dgp_name, hash(dgp_name) % 97)
    return n * 100 + iteration * 17 + dgp_offset


def _param_hash(params: dict) -> str:
    """Stable SHA-256 hex digest of sorted, canonical JSON of params."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _is_cached(data_path: Path, expected_hash: str) -> bool:
    """Check if parquet and its hash sidecar exist and match."""
    hash_path = data_path.with_suffix(".hash")
    if data_path.exists() and hash_path.exists():
        return hash_path.read_text().strip() == expected_hash
    return False


def _write_hash(data_path: Path, param_hash: str) -> None:
    data_path.with_suffix(".hash").write_text(param_hash)


def _base_dgp_n_obs(target_n: int, nb_year: int = 10) -> int:
    """Exact row count produced by `base_dgp` for target size `target_n`."""
    return round(target_n / nb_year) * nb_year


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
    total = burn_in + n_iters
    cached_count = 0

    for i in range(1, total + 1):
        iter_type = "burnin" if i <= burn_in else "iter"
        iter_num = i if i <= burn_in else i - burn_in
        dataset_id = f"{dgp_name}_{n}_{iter_type}_{iter_num}"
        data_path = data_dir / f"{dataset_id}.parquet"

        seed = _seed_for(dgp_name, n, i)
        params = {"dgp_type": dgp_type, "n": n, "seed": seed}
        h = _param_hash(params)
        n_obs_actual = _base_dgp_n_obs(n)

        if _is_cached(data_path, h):
            cached_count += 1
        else:
            df = base_dgp(n=n, type_=dgp_type, seed=seed)
            n_obs_actual = len(df)
            df.to_parquet(data_path)
            _write_hash(data_path, h)

        datasets.append(
            BenchmarkDataset(
                dataset_id=dataset_id,
                data_path=data_path.resolve(),
                dgp=dgp_name,
                n_obs=n_obs_actual,
                iter_type=iter_type,
                iter_num=iter_num,
            )
        )

    if cached_count == total:
        print(f"  [{dgp_name} n={n:,}] all {total} cached")
    elif cached_count > 0:
        print(f"  [{dgp_name} n={n:,}] {cached_count}/{total} cached, {total - cached_count} generated")

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


class BipartiteDGP:
    def __init__(
        self,
        data_dir: Path,
        n_time: int = 5,
        firm_size: int = 10,
        p_move: float = 0.5,
        c_sort: float = 1.0,
    ):
        self._data_dir = data_dir
        self._n_time = n_time
        self._firm_size = firm_size
        self._p_move = p_move
        self._c_sort = c_sort

    @property
    def dgp_name(self) -> str:
        return "bipartite"

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        datasets: list[BenchmarkDataset] = []
        n_workers = max(1, n // self._n_time)
        n_obs_actual = n_workers * self._n_time
        total = burn_in + n_iters
        cached_count = 0

        for i in range(1, total + 1):
            iter_type = "burnin" if i <= burn_in else "iter"
            iter_num = i if i <= burn_in else i - burn_in
            dataset_id = f"{self.dgp_name}_{n}_{iter_type}_{iter_num}"
            data_path = self._data_dir / f"{dataset_id}.parquet"

            seed = _seed_for(self.dgp_name, n, i)
            params = {
                "n_workers": n_workers,
                "n_time": self._n_time,
                "firm_size": self._firm_size,
                "p_move": self._p_move,
                "c_sort": self._c_sort,
                "seed": seed,
            }
            h = _param_hash(params)

            if _is_cached(data_path, h):
                cached_count += 1
            else:
                df = simulate_bipartite(
                    n_workers=n_workers,
                    n_time=self._n_time,
                    firm_size=self._firm_size,
                    p_move=self._p_move,
                    c_sort=self._c_sort,
                    seed=seed,
                )
                df.to_parquet(data_path)
                _write_hash(data_path, h)

            datasets.append(
                BenchmarkDataset(
                    dataset_id=dataset_id,
                    data_path=data_path.resolve(),
                    dgp=self.dgp_name,
                    n_obs=n_obs_actual,
                    iter_type=iter_type,
                    iter_num=iter_num,
                )
            )

        if cached_count == total:
            print(f"  [{self.dgp_name} n={n:,}] all {total} cached")
        elif cached_count > 0:
            print(f"  [{self.dgp_name} n={n:,}] {cached_count}/{total} cached, {total - cached_count} generated")

        return datasets
