from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

try:
    from .akm_dgp import AKMConfig, simulate_akm_panel
    from .dgp_functions import base_dgp
    from .interfaces import BenchmarkDataset
except ImportError:
    from akm_dgp import AKMConfig, simulate_akm_panel
    from dgp_functions import base_dgp
    from interfaces import BenchmarkDataset


def _seed_for(dgp_name: str, n: int, iteration: int) -> int:
    """Build deterministic seeds so benchmark runs are reproducible."""
    dgp_offset = {"simple": 0, "difficult": 1, "akm_baseline": 2}.get(
        dgp_name, hash(dgp_name) % 97
    )
    return n * 100 + iteration * 17 + dgp_offset


def _param_hash(params: dict) -> str:
    """Stable SHA-256 hex digest of sorted, canonical JSON of params."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _is_cached(data_path: Path, expected_hash: str) -> bool:
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


@dataclass(frozen=True)
class AKMSweepScenario:
    name: str
    overrides: dict[str, Any]


_AKM_DEFAULTS: dict[str, Any] = {
    "n_time": 10,
    "n_firms": 10_000,
    "n_industries": 5,
    "var_alpha": 1.0,
    "var_psi": 0.5,
    "var_phi": 0.1,
    "var_epsilon": 1.0,
    "gamma": 1.0,
    "rho_size": 0.6,
    "rho": 1.0,
    "delta": 0.2,
    "lambda_": 0.8,
    "beta_x1": 0.5,
    "n_match_bins": 64,
    "entry_exit_share": 0.0,
    "entry_exit_n_periods": 2,
}


def _scenario(name: str, **overrides: Any) -> AKMSweepScenario:
    return AKMSweepScenario(name=name, overrides=overrides)


def _expected_obs_per_worker(params: dict[str, Any]) -> float:
    n_time = int(params["n_time"])
    short_share = float(params.get("entry_exit_share", 0.0))
    short_periods = int(params.get("entry_exit_n_periods", n_time))
    return (1 - short_share) * n_time + short_share * short_periods


def _actual_obs_count(n_workers: int, params: dict[str, Any]) -> int:
    n_time = int(params["n_time"])
    short_share = float(params.get("entry_exit_share", 0.0))
    short_periods = int(params.get("entry_exit_n_periods", n_time))
    n_short_workers = int(round(short_share * n_workers))
    return (n_workers - n_short_workers) * n_time + n_short_workers * short_periods


def _infer_worker_count(target_n_obs: int, params: dict[str, Any]) -> int:
    expected_obs_per_worker = max(_expected_obs_per_worker(params), 1.0)
    floor_workers = max(1, int(target_n_obs / expected_obs_per_worker))
    candidates = {floor_workers, floor_workers + 1}
    return min(
        candidates,
        key=lambda n_workers: (
            abs(_actual_obs_count(n_workers, params) - target_n_obs),
            n_workers,
        ),
    )


def _akm_sweep_scenarios() -> list[AKMSweepScenario]:
    return [
        # ── Act 1: Reference points ──
        _scenario("akm_baseline"),
        _scenario("akm_easy", n_firms=100, delta=0.5, rho=0.0, n_time=20),
        # ── Act 2: Scale ──
        _scenario("akm_scale_1"),
        _scenario("akm_scale_2"),
        _scenario("akm_scale_3"),
        _scenario("akm_scale_4"),
        # ── Act 3: Single-axis sweeps (one knob, rest at defaults) ──
        # sorting (n_match_bins=2048 for near-continuous matching, delta=0.05
        # so fewer moves create fewer bridges between quality bands,
        # n_firms=50_000 so high rho creates many fine-grained quality bands)
        _scenario(
            "akm_sorting_1", rho=0.0, n_match_bins=2048, delta=0.05, n_firms=50_000
        ),
        _scenario(
            "akm_sorting_2", rho=5.0, n_match_bins=2048, delta=0.05, n_firms=50_000
        ),
        _scenario(
            "akm_sorting_3", rho=20.0, n_match_bins=2048, delta=0.05, n_firms=50_000
        ),
        _scenario(
            "akm_sorting_4", rho=50.0, n_match_bins=2048, delta=0.05, n_firms=50_000
        ),
        _scenario(
            "akm_sorting_5", rho=100.0, n_match_bins=2048, delta=0.05, n_firms=50_000
        ),
        # mobility
        _scenario("akm_mobility_1", delta=0.5),
        _scenario("akm_mobility_2", delta=0.05),
        _scenario("akm_mobility_3", delta=0.01),
        _scenario("akm_mobility_4", delta=0.005),
        _scenario("akm_mobility_5", delta=0.001),
        # firm size (n_firms=5000 so extreme skew creates genuinely tiny firms)
        _scenario("akm_size_1", gamma=100.0, n_firms=5_000),
        _scenario("akm_size_2", gamma=2.0, n_firms=5_000),
        _scenario("akm_size_3", gamma=0.5, n_firms=5_000),
        _scenario("akm_size_4", gamma=0.2, n_firms=5_000),
        # fragmentation (delta=0.05 so total movers are few, n_time=4 so
        # cross-industry bridges have few periods to accumulate)
        _scenario(
            "akm_fragmentation_1", n_industries=1, lambda_=1.0, delta=0.05, n_time=4
        ),
        _scenario(
            "akm_fragmentation_2", n_industries=5, lambda_=0.5, delta=0.05, n_time=4
        ),
        _scenario(
            "akm_fragmentation_3", n_industries=5, lambda_=0.95, delta=0.05, n_time=4
        ),
        _scenario(
            "akm_fragmentation_4", n_industries=20, lambda_=0.95, delta=0.05, n_time=4
        ),
        _scenario(
            "akm_fragmentation_5", n_industries=50, lambda_=0.99, delta=0.05, n_time=4
        ),
        # saturation (n_time=10, isolate FE-ratio effect)
        _scenario("akm_saturation_1", n_firms=1_000),
        _scenario("akm_saturation_2", n_firms=10_000),
        _scenario("akm_saturation_3", n_firms=50_000),
        _scenario("akm_saturation_4", n_firms=90_000),
        # short panels
        _scenario("akm_short_panel_1", n_time=2),
        _scenario("akm_short_panel_2", n_time=2, delta=0.1),
        _scenario("akm_short_panel_3", n_time=2, delta=0.05),
        _scenario("akm_short_panel_4", n_time=2, delta=0.02),
        # unbalanced panels (delta=0.05 so short-tenure workers are mostly
        # stayers contributing zero edges, straining the graph)
        _scenario(
            "akm_unbalanced_1",
            entry_exit_share=0.10,
            entry_exit_n_periods=2,
            delta=0.05,
        ),
        _scenario(
            "akm_unbalanced_2",
            entry_exit_share=0.25,
            entry_exit_n_periods=2,
            delta=0.05,
        ),
        _scenario(
            "akm_unbalanced_3",
            entry_exit_share=0.50,
            entry_exit_n_periods=2,
            delta=0.05,
        ),
        _scenario(
            "akm_unbalanced_4",
            entry_exit_share=0.75,
            entry_exit_n_periods=2,
            delta=0.05,
        ),
        # ── Act 4: Combinations ──
        # sorting x mobility (2x2 factorial)
        _scenario("akm_interaction_1", rho=0.0, delta=0.5),
        _scenario("akm_interaction_2", rho=20.0, delta=0.5),
        _scenario("akm_interaction_3", rho=0.0, delta=0.02),
        _scenario("akm_interaction_4", rho=20.0, delta=0.02),
        # fragmentation x low mobility (near-nested FE)
        _scenario("akm_nested_1", n_industries=100, lambda_=0.99, delta=0.01),
        _scenario("akm_nested_2", n_industries=50, lambda_=0.995, delta=0.005),
        # saturation x short panel
        _scenario(
            "akm_saturation_short_1",
            n_workers=500_000,
            n_firms=50_000,
            n_time=2,
        ),
        _scenario(
            "akm_saturation_short_2",
            n_workers=450_000,
            n_firms=400_000,
            n_time=2,
        ),
        # all levers combined
        _scenario(
            "akm_pathological_1",
            rho=50.0,
            delta=0.005,
            n_time=2,
        ),
        _scenario(
            "akm_pathological_2",
            rho=50.0,
            delta=0.005,
            n_industries=20,
            lambda_=0.95,
            n_time=2,
        ),
    ]


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


class AKMSweepDGP:
    def __init__(
        self,
        data_dir: Path,
        name: str,
        **overrides: Any,
    ):
        self._data_dir = data_dir
        self._name = name
        self._overrides = overrides

    @property
    def dgp_name(self) -> str:
        return self._name

    def _build_config(self, n: int = 1_000_000) -> AKMConfig:
        params = {**_AKM_DEFAULTS, **self._overrides}
        n_workers = int(params.pop("n_workers", _infer_worker_count(n, params)))
        return AKMConfig(n_workers=n_workers, **params)

    def generate(
        self, n: int, n_iters: int = 3, burn_in: int = 1
    ) -> list[BenchmarkDataset]:
        config = self._build_config(n=n)
        return _generate_datasets(
            dgp_name=self.dgp_name,
            n=n,
            n_iters=n_iters,
            burn_in=burn_in,
            data_dir=self._data_dir,
            make_params=lambda seed: {**asdict(config), "seed": seed},
            generate=lambda seed: simulate_akm_panel(config, seed=seed),
        )


def get_akm_sweep_scenarios(
    data_dir: Path, names: list[str] | None = None
) -> list[AKMSweepDGP]:
    scenario_defs = _akm_sweep_scenarios()
    scenario_map = {scenario.name: scenario for scenario in scenario_defs}
    scenario_names = names or [scenario.name for scenario in scenario_defs]
    unknown = sorted(set(scenario_names) - set(scenario_map))
    if unknown:
        raise ValueError(f"Unknown AKM sweep scenario(s): {', '.join(unknown)}")

    return [
        AKMSweepDGP(
            data_dir=data_dir,
            name=scenario_map[name].name,
            **scenario_map[name].overrides,
        )
        for name in scenario_names
    ]
