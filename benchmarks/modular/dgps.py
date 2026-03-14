from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

try:
    from .dgp_functions import BipartiteConfig, base_dgp, simulate_bipartite
    from .interfaces import BenchmarkDataset
except ImportError:
    from dgp_functions import BipartiteConfig, base_dgp, simulate_bipartite
    from interfaces import BenchmarkDataset


def _seed_for(dgp_name: str, n: int, iteration: int) -> int:
    """Build deterministic seeds so benchmark runs are reproducible."""
    dgp_offset = {"simple": 0, "difficult": 1, "bipartite": 2}.get(
        dgp_name, hash(dgp_name) % 97
    )
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


def _generate_datasets(
    *,
    dgp_name: str,
    n: int,
    n_iters: int,
    burn_in: int,
    data_dir: Path,
    make_params: Callable[[int], dict],
    generate: Callable[[int], Any],
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
        params = make_params(seed)
        h = _param_hash(params)

        if _is_cached(data_path, h):
            cached_count += 1
            n_obs_actual = pq.read_metadata(data_path).num_rows
        else:
            df = generate(seed)
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
        print(
            f"  [{dgp_name} n={n:,}] {cached_count}/{total} cached, {total - cached_count} generated"
        )

    return datasets


class BaseDGP:
    def __init__(self, data_dir: Path, dgp_type: str = "simple"):
        self._data_dir = data_dir
        self._dgp_type = dgp_type

    @property
    def dgp_name(self) -> str:
        return self._dgp_type

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        dgp_type = self._dgp_type
        return _generate_datasets(
            dgp_name=self.dgp_name,
            n=n,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
            make_params=lambda seed: {"dgp_type": dgp_type, "n": n, "seed": seed},
            generate=lambda seed: base_dgp(n=n, type_=dgp_type, seed=seed),
        )


class BipartiteDGP:
    def __init__(
        self,
        data_dir: Path,
        name: str = "bipartite",
        n_time: int = 5,
        firm_size: int = 10,
        n_firm_types: int = 5,
        n_worker_types: int = 5,
        p_move: float = 0.5,
        c_sort: float = 1.0,
        n_clusters: int = 1,
        cross_cluster_scale: float = 1.0,
        firm_size_dist: str = "equal",
        firm_size_lognorm_sigma: float = 1.0,
        firm_size_pareto_shape: float = 1.5,
    ):
        self._name = name
        self._data_dir = data_dir
        self._n_time = n_time
        self._firm_size = firm_size
        self._n_firm_types = n_firm_types
        self._n_worker_types = n_worker_types
        self._p_move = p_move
        self._c_sort = c_sort
        self._n_clusters = n_clusters
        self._cross_cluster_scale = cross_cluster_scale
        self._firm_size_dist = firm_size_dist
        self._firm_size_lognorm_sigma = firm_size_lognorm_sigma
        self._firm_size_pareto_shape = firm_size_pareto_shape

    @property
    def dgp_name(self) -> str:
        return self._name

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        n_workers = max(1, n // self._n_time)
        config = BipartiteConfig(
            n_workers=n_workers,
            n_time=self._n_time,
            firm_size=self._firm_size,
            n_firm_types=self._n_firm_types,
            n_worker_types=self._n_worker_types,
            p_move=self._p_move,
            c_sort=self._c_sort,
            n_clusters=self._n_clusters,
            cross_cluster_scale=self._cross_cluster_scale,
            firm_size_dist=self._firm_size_dist,
            firm_size_lognorm_sigma=self._firm_size_lognorm_sigma,
            firm_size_pareto_shape=self._firm_size_pareto_shape,
        )
        return _generate_datasets(
            dgp_name=self.dgp_name,
            n=n,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
            make_params=lambda seed: {**asdict(config), "seed": seed},
            generate=lambda seed: simulate_bipartite(config, seed=seed),
        )
