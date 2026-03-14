from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def base_dgp(
    n=1000,
    nb_year=10,
    nb_indiv_per_firm=23,
    type_="simple",
    seed=None,
):
    """Generate panel data with individual, firm, and year fixed effects.

    Parameters
    ----------
    n : int
        Total number of observations (approximately).
    nb_year : int
        Number of time periods.
    nb_indiv_per_firm : int
        Average number of individuals per firm.
    type_ : str
        Either "simple" (random firm assignment) or "difficult" (deterministic
        cycling firm assignment).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: indiv_id, firm_id, year, x1, x2, y, exp_y, negbin_y, binary_y.
    """
    rng = np.random.default_rng(seed)

    nb_indiv = round(n / nb_year)
    nb_firm = round(nb_indiv / nb_indiv_per_firm)

    if nb_indiv < 1 or nb_firm < 1:
        raise ValueError(
            f"n={n} too small for nb_year={nb_year}, "
            f"nb_indiv_per_firm={nb_indiv_per_firm}"
        )

    # Actual observation count (may differ from n when n % nb_year != 0)
    n_obs = nb_indiv * nb_year

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n_obs)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n_obs // nb_firm + 1)[:n_obs]
    else:
        raise ValueError(f"Unknown type of dgp: {type_!r}")

    x1 = rng.standard_normal(n_obs)
    x2 = x1**2

    firm_fe = rng.standard_normal(nb_firm)[firm_id - 1]
    unit_fe = rng.standard_normal(nb_indiv)[indiv_id - 1]
    year_fe = rng.standard_normal(nb_year)[year - 1]
    mu = 1 * x1 + 0.05 * x2 + firm_fe + unit_fe + year_fe
    y = mu + rng.standard_normal(len(mu))

    # Negative binomial: R's rnegbin(mu, theta) parameterization
    # n = theta, p = theta / (theta + mu)
    theta = 0.5
    exp_y = np.exp(y)
    nb_p = theta / (theta + exp_y)
    negbin_y = rng.negative_binomial(n=theta, p=nb_p)

    return pd.DataFrame(
        {
            "indiv_id": indiv_id,
            "firm_id": firm_id,
            "year": year,
            "x1": x1,
            "x2": x2,
            "y": y,
            "exp_y": exp_y,
            "negbin_y": negbin_y,
            "binary_y": (y > 0).astype(int),
        }
    )


@dataclass(frozen=True)
class BipartiteConfig:
    """Configuration for bipartite labor market simulation (excludes seed)."""

    n_workers: int = 10_000
    n_time: int = 5
    firm_size: int = 10
    n_firm_types: int = 5
    n_worker_types: int = 5
    p_move: float = 0.5
    c_sort: float = 1.0
    c_netw: float = 1.0
    c_sig: float = 1.0
    n_clusters: int = 1
    cross_cluster_scale: float = 1.0
    firm_size_dist: str = "equal"
    firm_size_lognorm_sigma: float = 1.0
    firm_size_pareto_shape: float = 1.5
    alpha_sig: float = 1.0
    psi_sig: float = 1.0
    w_sig: float = 1.0
    x_sig: float = 1.0
    y1_beta: float = 0.5
    y1_sig: float = 1.0
    y2_beta: float = 0.3
    y2_sig: float = 1.0


def _validate_config(config: BipartiteConfig):
    """Validate bipartite simulation parameters."""
    if config.n_workers < 1:
        raise ValueError("n_workers must be positive")
    if config.n_time < 1:
        raise ValueError("n_time must be positive")
    if config.firm_size <= 0:
        raise ValueError("firm_size must be positive")
    if config.n_firm_types < 1:
        raise ValueError("n_firm_types must be positive")
    if config.n_worker_types < 1:
        raise ValueError("n_worker_types must be positive")
    if config.n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if config.n_clusters > min(config.n_firm_types, config.n_worker_types):
        raise ValueError("n_clusters must not exceed the number of types")
    if not 0 <= config.p_move <= 1:
        raise ValueError("p_move must be in [0, 1]")
    if config.cross_cluster_scale <= 0:
        raise ValueError("cross_cluster_scale must be positive")
    if config.firm_size_dist not in {"equal", "lognormal", "pareto"}:
        raise ValueError("firm_size_dist must be one of: equal, lognormal, pareto")
    if config.firm_size_lognorm_sigma <= 0:
        raise ValueError("firm_size_lognorm_sigma must be positive")
    if config.firm_size_pareto_shape <= 0:
        raise ValueError("firm_size_pareto_shape must be positive")
    for name, val in [
        ("alpha_sig", config.alpha_sig),
        ("psi_sig", config.psi_sig),
        ("w_sig", config.w_sig),
        ("c_sig", config.c_sig),
        ("x_sig", config.x_sig),
        ("y1_sig", config.y1_sig),
        ("y2_sig", config.y2_sig),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative")


def _cluster_codes(n_types: int, n_clusters: int) -> np.ndarray:
    """Assign contiguous type indices to a small number of coarse clusters."""
    return np.minimum((np.arange(n_types) * n_clusters) // n_types, n_clusters - 1)


def _generate_fixed_effects(config: BipartiteConfig):
    """Generate firm (psi) and worker (alpha) fixed effects from quantiles."""
    psi = (
        norm.ppf(np.arange(1, config.n_firm_types + 1) / (config.n_firm_types + 1))
        * config.psi_sig
    )
    alpha = (
        norm.ppf(np.arange(1, config.n_worker_types + 1) / (config.n_worker_types + 1))
        * config.alpha_sig
    )
    return psi, alpha


def _compute_transition_matrices(config: BipartiteConfig, psi, alpha):
    """Compute worker-type-specific firm-to-firm transition matrices."""
    G = np.zeros((config.n_worker_types, config.n_firm_types, config.n_firm_types))
    firm_clusters = _cluster_codes(config.n_firm_types, config.n_clusters)
    for typ_no in range(config.n_worker_types):
        for k_from in range(config.n_firm_types):
            probs = norm.pdf(
                (psi - config.c_netw * psi[k_from] - config.c_sort * alpha[typ_no])
                / config.c_sig
            )
            if config.n_clusters > 1 and config.cross_cluster_scale != 1.0:
                cross_cluster = firm_clusters != firm_clusters[k_from]
                probs = probs.copy()
                probs[cross_cluster] *= config.cross_cluster_scale
            G[typ_no, k_from, :] = probs / probs.sum()
    return G


def _compute_stationary_distributions(G):
    """Compute stationary distribution for each worker type's transition matrix."""
    n_worker_types = G.shape[0]
    n_firm_types = G.shape[1]
    H = np.zeros((n_worker_types, n_firm_types))
    for typ_no in range(n_worker_types):
        eigvals, eigvecs = np.linalg.eig(G[typ_no].T)
        stationary_idx = np.argmin(np.abs(eigvals - 1))
        stationary_vec = np.real(eigvecs[:, stationary_idx])
        stationary_vec = np.abs(stationary_vec) / np.abs(stationary_vec).sum()
        H[typ_no] = stationary_vec
    return H


def _simulate_mobility(rng, config: BipartiteConfig, H, G, worker_types):
    """Simulate worker mobility across firms over time (vectorized over workers)."""
    n_workers, n_time = config.n_workers, config.n_time
    firm_types = np.zeros((n_workers, n_time), dtype=int)

    # Vectorized initial assignment by worker type
    for typ_no in range(config.n_worker_types):
        mask = worker_types == typ_no
        firm_types[mask, 0] = rng.choice(
            config.n_firm_types, size=mask.sum(), p=H[typ_no]
        )

    # Pre-generate all move decisions at once
    move_decisions = rng.random((n_workers, n_time - 1)) < config.p_move

    # Vectorized time evolution
    for t in range(1, n_time):
        movers = move_decisions[:, t - 1]
        firm_types[~movers, t] = firm_types[~movers, t - 1]  # stayers
        for typ_no in range(config.n_worker_types):
            for k_from in range(config.n_firm_types):
                mask = (
                    movers & (worker_types == typ_no) & (firm_types[:, t - 1] == k_from)
                )
                n = mask.sum()
                if n > 0:
                    firm_types[mask, t] = rng.choice(
                        config.n_firm_types, size=n, p=G[typ_no, k_from]
                    )

    # Vectorized spell IDs via cumulative sum
    new_spell = np.ones((n_workers, n_time), dtype=bool)
    new_spell[:, 1:] = move_decisions
    spell_ids = np.cumsum(new_spell.ravel()).reshape(n_workers, n_time) - 1
    spell_counter = int(spell_ids.max()) + 1

    return firm_types, spell_ids, spell_counter


def _firm_assignment_weights(rng, config: BipartiteConfig, n_firms: int) -> np.ndarray:
    """Build spell-to-firm assignment weights for one firm type."""
    if config.firm_size_dist == "equal" or n_firms == 1:
        weights = np.ones(n_firms)
    elif config.firm_size_dist == "lognormal":
        weights = rng.lognormal(
            mean=0.0, sigma=config.firm_size_lognorm_sigma, size=n_firms
        )
    else:
        weights = rng.pareto(config.firm_size_pareto_shape, size=n_firms) + 1.0
    return weights / weights.sum()


def _assign_firm_ids(
    rng, config: BipartiteConfig, firm_types, spell_ids, spell_counter
):
    """Assign unique firm IDs per spell using vectorized numpy operations."""
    firm_type_flat = firm_types.ravel()
    spell_flat = spell_ids.ravel()
    spell_to_firm_type = np.empty(spell_counter, dtype=int)
    spell_to_firm_type[spell_flat] = firm_type_flat
    spell_sizes = np.bincount(spell_flat, minlength=spell_counter)

    firm_id_per_spell = np.zeros(spell_counter, dtype=int)
    global_offset = 0
    for k_type in np.unique(spell_to_firm_type):
        k_mask = spell_to_firm_type == k_type
        total_obs = spell_sizes[k_mask].sum()
        n_firms = max(1, round(total_obs / (config.firm_size * config.n_time)))
        weights = _firm_assignment_weights(rng, config, n_firms)
        assigned = rng.choice(n_firms, size=k_mask.sum(), p=weights)
        _, remapped = np.unique(assigned, return_inverse=True)
        firm_id_per_spell[k_mask] = remapped + global_offset
        global_offset += len(np.unique(assigned))

    return firm_id_per_spell[spell_flat]


def _build_dataframe(
    rng, config: BipartiteConfig, worker_types, firm_types, firm_id, psi, alpha
):
    """Build the final DataFrame with outcomes."""
    n_workers, n_time = config.n_workers, config.n_time
    n_obs = n_workers * n_time
    worker_clusters = _cluster_codes(config.n_worker_types, config.n_clusters)
    firm_clusters = _cluster_codes(config.n_firm_types, config.n_clusters)

    indiv_id = np.repeat(np.arange(n_workers), n_time)
    time = np.tile(np.arange(n_time), n_workers)
    worker_type = np.repeat(worker_types, n_time)
    firm_type_flat = firm_types.ravel()

    worker_fe = alpha[worker_type]
    firm_fe = psi[firm_type_flat]
    wage = worker_fe + firm_fe + rng.standard_normal(n_obs) * config.w_sig
    x1 = rng.standard_normal(n_obs) * config.x_sig
    y = (
        worker_fe
        + firm_fe
        + config.y1_beta * x1
        + rng.standard_normal(n_obs) * config.y1_sig
    )
    y2 = (
        worker_fe
        + firm_fe
        + config.y2_beta * x1
        + rng.standard_normal(n_obs) * config.y2_sig
    )

    # Already sorted by (indiv_id, year) due to repeat/tile construction
    return pd.DataFrame(
        {
            "indiv_id": indiv_id,
            "firm_id": firm_id,
            "wage": wage,
            "y": y,
            "y2": y2,
            "x1": x1,
            "year": time,
            "worker_type": worker_type,
            "firm_type": firm_type_flat,
            "worker_cluster": worker_clusters[worker_type],
            "firm_cluster": firm_clusters[firm_type_flat],
            "worker_fe": worker_fe,
            "firm_fe": firm_fe,
        }
    )


def simulate_bipartite(config: BipartiteConfig, *, seed: int | None = None):
    """Simulate a bipartite labor market network.

    Parameters
    ----------
    config : BipartiteConfig
        Simulation configuration.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: indiv_id, firm_id, wage, y, y2, x1, year, worker_type,
        firm_type, worker_fe, firm_fe.
    """
    rng = np.random.default_rng(seed)
    _validate_config(config)
    psi, alpha = _generate_fixed_effects(config)
    G = _compute_transition_matrices(config, psi, alpha)
    H = _compute_stationary_distributions(G)
    worker_types = rng.integers(0, config.n_worker_types, size=config.n_workers)
    firm_types, spell_ids, spell_counter = _simulate_mobility(
        rng, config, H, G, worker_types
    )
    firm_id = _assign_firm_ids(rng, config, firm_types, spell_ids, spell_counter)
    return _build_dataframe(rng, config, worker_types, firm_types, firm_id, psi, alpha)
