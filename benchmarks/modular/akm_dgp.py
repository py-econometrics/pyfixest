from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class AKMConfig:
    """Configuration for the standalone AKM benchmark DGP."""

    n_workers: int = 100_000
    n_firms: int = 10_000
    n_time: int = 10
    n_industries: int = 5
    var_alpha: float = 1.0
    var_psi: float = 0.5
    var_phi: float = 0.1
    var_epsilon: float = 1.0
    gamma: float = 1.0
    rho_size: float = 0.6
    rho: float = 1.0
    delta: float | tuple[float, ...] = 0.2
    lambda_: float = 0.8
    beta_x1: float = 0.5
    n_match_bins: int = 64
    entry_exit_share: float = 0.0
    entry_exit_n_periods: int = 2


def _validate_config(config: AKMConfig) -> None:
    if config.n_workers < 1:
        raise ValueError("n_workers must be positive")
    if config.n_firms < 1:
        raise ValueError("n_firms must be positive")
    if config.n_time < 2:
        raise ValueError("n_time must be at least 2")
    if config.n_industries < 1:
        raise ValueError("n_industries must be positive")
    if config.n_industries > config.n_firms:
        raise ValueError("n_industries must not exceed n_firms")
    if config.n_match_bins < 1:
        raise ValueError("n_match_bins must be positive")
    if not 0 <= config.entry_exit_share <= 1:
        raise ValueError("entry_exit_share must be in [0, 1]")
    if not 2 <= config.entry_exit_n_periods <= config.n_time:
        raise ValueError("entry_exit_n_periods must be in [2, n_time]")
    if config.gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 0 <= config.rho_size <= 1:
        raise ValueError("rho_size must be in [0, 1]")
    if config.rho < 0:
        raise ValueError("rho must be non-negative")
    if isinstance(config.delta, tuple):
        if len(config.delta) != config.n_industries:
            raise ValueError(
                f"delta tuple length {len(config.delta)} != n_industries {config.n_industries}"
            )
        if not all(0 < d <= 1 for d in config.delta):
            raise ValueError("all delta values must be in (0, 1]")
    else:
        if not 0 < config.delta <= 1:
            raise ValueError("delta must be in (0, 1]")
    if config.n_industries == 1:
        if config.lambda_ != 1:
            raise ValueError("lambda_ must equal 1 when n_industries == 1")
    elif not (1 / config.n_industries) <= config.lambda_ <= 1:
        raise ValueError("lambda_ must be in [1 / n_industries, 1]")
    for name, value in (
        ("var_alpha", config.var_alpha),
        ("var_psi", config.var_psi),
        ("var_phi", config.var_phi),
        ("var_epsilon", config.var_epsilon),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _balanced_groups(
    n_items: int, n_groups: int, rng: np.random.Generator
) -> np.ndarray:
    """Assign groups nearly uniformly while guaranteeing support for every group."""
    groups = np.arange(n_items) % n_groups
    return rng.permutation(groups)


def _couple_by_rank(
    rng: np.random.Generator,
    left: np.ndarray,
    right: np.ndarray,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign sorted draws to correlated latent ranks via a Gaussian copula."""
    z1 = rng.standard_normal(len(left))
    z2 = correlation * z1 + np.sqrt(max(0.0, 1 - correlation**2)) * rng.standard_normal(
        len(right)
    )
    left_order = np.argsort(z1)
    right_order = np.argsort(z2)

    left_out = np.empty_like(left)
    right_out = np.empty_like(right)
    left_out[left_order] = np.sort(left)
    right_out[right_order] = np.sort(right)
    return left_out, right_out


def _firm_size_weights(config: AKMConfig, rng: np.random.Generator) -> np.ndarray:
    u = np.clip(rng.random(config.n_firms), 1e-12, 1 - 1e-12)
    raw = np.exp(-np.log(u) / config.gamma)
    return raw / raw.sum()


def _alpha_bins(alpha: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucket workers by alpha rank for fast approximate sampling."""
    n_bins = min(n_bins, len(alpha))
    order = np.argsort(alpha, kind="mergesort")
    bin_ids = np.empty(len(alpha), dtype=np.int16)
    bin_ids[order] = (np.arange(len(alpha)) * n_bins) // len(alpha)
    centers = np.empty(n_bins, dtype=float)
    for bin_id in range(n_bins):
        centers[bin_id] = alpha[bin_ids == bin_id].mean()
    return bin_ids, centers


def _industry_weights(config: AKMConfig, firm_industries: np.ndarray) -> np.ndarray:
    weights = np.full((config.n_industries, config.n_firms), 1.0, dtype=float)
    if config.n_industries == 1:
        return weights

    outside_weight = (1 - config.lambda_) / (config.n_industries - 1)
    weights.fill(outside_weight)
    for industry in range(config.n_industries):
        weights[industry, firm_industries == industry] = config.lambda_
    return weights


def _build_assignment_cdfs(
    config: AKMConfig,
    alpha_centers: np.ndarray,
    psi: np.ndarray,
    firm_weights: np.ndarray,
    firm_industries: np.ndarray,
) -> np.ndarray:
    tau2 = max(config.var_alpha + config.var_psi, 1e-12)
    industry_weights = _industry_weights(config, firm_industries)

    cdfs = np.empty(
        (config.n_industries, len(alpha_centers), config.n_firms), dtype=np.float32
    )
    for industry in range(config.n_industries):
        for bin_id, alpha_center in enumerate(alpha_centers):
            scores = (
                np.exp(-0.5 * config.rho * ((alpha_center - psi) ** 2) / tau2)
                * firm_weights
                * industry_weights[industry]
            )
            probs = scores / scores.sum()
            cdf = np.cumsum(probs, dtype=np.float64)
            cdf[-1] = 1.0
            cdfs[industry, bin_id] = cdf.astype(np.float32)

    return cdfs


def _group_worker_indices(
    worker_industries: np.ndarray,
    worker_bins: np.ndarray,
    n_industries: int,
    n_bins: int,
) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], np.ndarray] = {}
    for industry in range(n_industries):
        industry_mask = worker_industries == industry
        for bin_id in range(n_bins):
            idx = np.flatnonzero(industry_mask & (worker_bins == bin_id))
            if idx.size:
                groups[(industry, bin_id)] = idx
    return groups


def _sample_firms(
    rng: np.random.Generator,
    cdf: np.ndarray,
    size: int,
    current: np.ndarray | None = None,
) -> np.ndarray:
    draws = np.searchsorted(cdf, rng.random(size), side="right")
    if current is None or len(cdf) == 1:
        return draws

    same = draws == current
    attempts = 0
    while same.any() and attempts < 8:
        draws[same] = np.searchsorted(cdf, rng.random(same.sum()), side="right")
        same = draws == current
        attempts += 1

    if same.any():
        draws[same] = (current[same] + 1) % len(cdf)

    return draws


def _panel_observation_mask(
    config: AKMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones((config.n_workers, config.n_time), dtype=bool)
    if config.entry_exit_share == 0 or config.entry_exit_n_periods == config.n_time:
        return mask

    n_short = int(round(config.entry_exit_share * config.n_workers))
    if n_short == 0:
        return mask

    short_workers = rng.choice(config.n_workers, size=n_short, replace=False)
    starts = rng.integers(
        0,
        config.n_time - config.entry_exit_n_periods + 1,
        size=n_short,
    )
    periods = np.arange(config.n_time)
    short_mask = (periods[None, :] >= starts[:, None]) & (
        periods[None, :] < starts[:, None] + config.entry_exit_n_periods
    )
    mask[short_workers] = short_mask
    return mask


def simulate_akm_panel(
    config: AKMConfig,
    *,
    seed: int | None = None,
    include_latent: bool = False,
) -> pd.DataFrame:
    """Simulate an AKM-style worker-firm panel for benchmark timing."""
    _validate_config(config)
    rng = np.random.default_rng(seed)

    raw_psi = rng.normal(scale=np.sqrt(config.var_psi), size=config.n_firms)
    raw_size = _firm_size_weights(config, rng)
    psi, firm_weights = _couple_by_rank(rng, raw_psi, raw_size, config.rho_size)
    firm_weights = firm_weights / firm_weights.sum()

    firm_industries = _balanced_groups(config.n_firms, config.n_industries, rng)
    alpha = rng.normal(scale=np.sqrt(config.var_alpha), size=config.n_workers)
    worker_industries = rng.integers(0, config.n_industries, size=config.n_workers)
    worker_bins, alpha_centers = _alpha_bins(alpha, config.n_match_bins)
    cdfs = _build_assignment_cdfs(
        config,
        alpha_centers,
        psi,
        firm_weights,
        firm_industries,
    )
    worker_groups = _group_worker_indices(
        worker_industries,
        worker_bins,
        config.n_industries,
        len(alpha_centers),
    )

    firm_paths = np.empty((config.n_workers, config.n_time), dtype=np.int32)
    for (industry, bin_id), idx in worker_groups.items():
        firm_paths[idx, 0] = _sample_firms(rng, cdfs[industry, bin_id], len(idx))

    if isinstance(config.delta, tuple):
        worker_deltas = np.array(config.delta)[worker_industries]  # (n_workers,)
        move_draws = (
            rng.random((config.n_workers, config.n_time - 1)) < worker_deltas[:, None]
        )
    else:
        move_draws = rng.random((config.n_workers, config.n_time - 1)) < config.delta
    for t in range(1, config.n_time):
        firm_paths[:, t] = firm_paths[:, t - 1]
        movers = move_draws[:, t - 1]
        for (industry, bin_id), idx in worker_groups.items():
            move_idx = idx[movers[idx]]
            if move_idx.size == 0:
                continue
            current = firm_paths[move_idx, t - 1]
            firm_paths[move_idx, t] = _sample_firms(
                rng,
                cdfs[industry, bin_id],
                len(move_idx),
                current=current,
            )

    observed_mask = _panel_observation_mask(config, rng).ravel()

    n_full_obs = config.n_workers * config.n_time
    indiv_id = np.repeat(np.arange(1, config.n_workers + 1), config.n_time)
    year = np.tile(np.arange(1, config.n_time + 1), config.n_workers)
    firm_id = firm_paths.ravel() + 1
    x1 = rng.standard_normal(n_full_obs)
    year_fe_values = rng.normal(scale=np.sqrt(config.var_phi), size=config.n_time)
    year_fe = np.tile(year_fe_values, config.n_workers)
    worker_fe = np.repeat(alpha, config.n_time)
    firm_fe = psi[firm_paths.ravel()]
    epsilon = rng.normal(scale=np.sqrt(config.var_epsilon), size=n_full_obs)
    y = worker_fe + firm_fe + year_fe + config.beta_x1 * x1 + epsilon

    data = {
        "indiv_id": indiv_id[observed_mask],
        "firm_id": firm_id[observed_mask],
        "year": year[observed_mask],
        "x1": x1[observed_mask],
        "y": y[observed_mask],
    }
    if include_latent:
        data["worker_industry"] = np.repeat(worker_industries, config.n_time)[
            observed_mask
        ]
        data["firm_industry"] = firm_industries[firm_paths.ravel()][observed_mask]
        data["worker_fe"] = worker_fe[observed_mask]
        data["firm_fe"] = firm_fe[observed_mask]
        data["year_fe"] = year_fe[observed_mask]

    return pd.DataFrame(data)


def summarize_akm_panel(df: pd.DataFrame) -> dict[str, float]:
    """Compute graph and mobility diagnostics from a simulated AKM panel."""
    obs_per_worker = df.groupby("indiv_id").size()
    mover_worker_ids = (
        df.groupby("indiv_id")["firm_id"]
        .nunique()
        .loc[lambda s: s > 1]
        .index.to_numpy()
    )
    mover_obs = df[df["indiv_id"].isin(mover_worker_ids)]
    edges = mover_obs[["indiv_id", "firm_id"]].drop_duplicates()

    diagnostics: dict[str, float] = {
        "mean_observed_periods": float(obs_per_worker.mean()),
        "two_period_worker_share": float(obs_per_worker.eq(2).mean()),
        "mover_share": float(len(mover_worker_ids) / df["indiv_id"].nunique()),
        "singleton_count": float(
            df.groupby(["indiv_id", "firm_id"]).size().eq(1).sum()
        ),
    }

    if "firm_industry" in df.columns:
        diagnostics["cross_industry_share"] = float(
            df.groupby("indiv_id")["firm_industry"].nunique().gt(1).mean()
        )

    if edges.empty:
        diagnostics["connected_components"] = 0.0
        diagnostics["largest_connected_set_share"] = 0.0
        return diagnostics

    worker_codes, worker_levels = pd.factorize(edges["indiv_id"], sort=False)
    firm_codes, firm_levels = pd.factorize(edges["firm_id"], sort=False)
    n_workers = len(worker_levels)
    row = np.concatenate([worker_codes, firm_codes + n_workers])
    col = np.concatenate([firm_codes + n_workers, worker_codes])
    data = np.ones(len(row), dtype=np.int8)
    graph = csr_matrix((data, (row, col)), shape=(n_workers + len(firm_levels),) * 2)
    n_components, labels = connected_components(graph, directed=False)
    component_sizes = np.bincount(labels)
    largest_component = int(component_sizes.argmax())

    worker_component = pd.Series(labels[:n_workers], index=worker_levels)
    firm_component = pd.Series(labels[n_workers:], index=firm_levels)
    in_largest = df["indiv_id"].map(worker_component).eq(largest_component) & df[
        "firm_id"
    ].map(firm_component).eq(largest_component)

    diagnostics["connected_components"] = float(n_components)
    diagnostics["largest_connected_set_share"] = float(in_largest.mean())
    return diagnostics
