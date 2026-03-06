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
    alpha_sig: float = 1.0
    psi_sig: float = 1.0
    w_sig: float = 1.0
    x_sig: float = 1.0
    y1_beta: float = 0.5
    y1_sig: float = 1.0
    y2_beta: float = 0.3
    y2_sig: float = 1.0


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

    # Unpack config
    n_workers = config.n_workers
    n_time = config.n_time
    firm_size = config.firm_size
    n_firm_types = config.n_firm_types
    n_worker_types = config.n_worker_types
    p_move = config.p_move
    c_sort = config.c_sort
    c_netw = config.c_netw
    c_sig = config.c_sig
    alpha_sig = config.alpha_sig
    psi_sig = config.psi_sig
    w_sig = config.w_sig
    x_sig = config.x_sig
    y1_beta = config.y1_beta
    y1_sig = config.y1_sig
    y2_beta = config.y2_beta
    y2_sig = config.y2_sig

    # Parameter validation
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if n_time < 1:
        raise ValueError("n_time must be positive")
    if firm_size <= 0:
        raise ValueError("firm_size must be positive")
    if n_firm_types < 1:
        raise ValueError("n_firm_types must be positive")
    if n_worker_types < 1:
        raise ValueError("n_worker_types must be positive")
    if not 0 <= p_move <= 1:
        raise ValueError("p_move must be in [0, 1]")
    for name, val in [
        ("alpha_sig", alpha_sig),
        ("psi_sig", psi_sig),
        ("w_sig", w_sig),
        ("c_sig", c_sig),
        ("x_sig", x_sig),
        ("y1_sig", y1_sig),
        ("y2_sig", y2_sig),
    ]:
        if val < 0:
            raise ValueError(f"{name} must be non-negative")

    # ========================================================================
    # GENERATE FIXED EFFECTS
    # ========================================================================
    psi = norm.ppf(np.arange(1, n_firm_types + 1) / (n_firm_types + 1)) * psi_sig
    alpha = (
        norm.ppf(np.arange(1, n_worker_types + 1) / (n_worker_types + 1)) * alpha_sig
    )

    # Compute transition matrices
    G = np.zeros((n_worker_types, n_firm_types, n_firm_types))
    for typ_no in range(n_worker_types):
        for k_from in range(n_firm_types):
            probs = norm.pdf(
                (psi - c_netw * psi[k_from] - c_sort * alpha[typ_no]) / c_sig
            )
            G[typ_no, k_from, :] = probs / probs.sum()

    # Compute stationary distributions
    H = np.zeros((n_worker_types, n_firm_types))
    for typ_no in range(n_worker_types):
        eigvals, eigvecs = np.linalg.eig(G[typ_no].T)
        stationary_idx = np.argmin(np.abs(eigvals - 1))
        stationary_vec = np.real(eigvecs[:, stationary_idx])
        stationary_vec = np.abs(stationary_vec) / np.abs(stationary_vec).sum()
        H[typ_no] = stationary_vec

    # ========================================================================
    # SIMULATE MOBILITY
    # ========================================================================
    worker_types = rng.integers(0, n_worker_types, size=n_workers)

    firm_types = np.zeros((n_workers, n_time), dtype=int)
    spell_ids = np.zeros((n_workers, n_time), dtype=int)
    spell_counter = 0

    for i in range(n_workers):
        typ_no = worker_types[i]

        firm_types[i, 0] = rng.choice(n_firm_types, p=H[typ_no])
        spell_ids[i, 0] = spell_counter
        spell_counter += 1

        for t in range(1, n_time):
            if rng.random() < p_move:
                firm_types[i, t] = rng.choice(
                    n_firm_types, p=G[typ_no, firm_types[i, t - 1]]
                )
                spell_ids[i, t] = spell_counter
                spell_counter += 1
            else:
                firm_types[i, t] = firm_types[i, t - 1]
                spell_ids[i, t] = spell_ids[i, t - 1]

    # ========================================================================
    # ASSIGN FIRM IDS (numpy-based, avoids intermediate DataFrame)
    # ========================================================================
    indiv_id = np.repeat(np.arange(n_workers), n_time)
    time = np.tile(np.arange(n_time), n_workers)
    worker_type = np.repeat(worker_types, n_time)
    firm_type_flat = firm_types.ravel()
    spell_flat = spell_ids.ravel()

    spell_to_firm_type = np.empty(spell_counter, dtype=int)
    spell_to_firm_type[spell_flat] = firm_type_flat
    spell_sizes = np.bincount(spell_flat, minlength=spell_counter)

    firm_id_per_spell = np.zeros(spell_counter, dtype=int)
    for k_type in np.unique(spell_to_firm_type):
        k_mask = spell_to_firm_type == k_type
        total_obs = spell_sizes[k_mask].sum()
        n_firms = max(1, round(total_obs / (firm_size * n_time)))
        assigned = rng.integers(0, n_firms, size=k_mask.sum())
        unique_assigned = np.unique(assigned)
        mapping = {v: idx for idx, v in enumerate(unique_assigned)}
        firm_id_per_spell[k_mask] = np.array([mapping[a] for a in assigned])

    firm_id = firm_id_per_spell[spell_flat]

    # ========================================================================
    # GENERATE OUTCOMES
    # ========================================================================
    n_obs = n_workers * n_time
    worker_fe = alpha[worker_type]
    firm_fe = psi[firm_type_flat]
    wage = worker_fe + firm_fe + rng.standard_normal(n_obs) * w_sig
    x1 = rng.standard_normal(n_obs) * x_sig
    y = worker_fe + firm_fe + y1_beta * x1 + rng.standard_normal(n_obs) * y1_sig
    y2 = worker_fe + firm_fe + y2_beta * x1 + rng.standard_normal(n_obs) * y2_sig

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
            "worker_fe": worker_fe,
            "firm_fe": firm_fe,
        }
    )
