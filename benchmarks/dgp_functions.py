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

    indiv_id = np.repeat(np.arange(1, nb_indiv + 1), nb_year)
    year = np.tile(np.arange(1, nb_year + 1), nb_indiv)

    if type_ == "simple":
        firm_id = rng.integers(1, nb_firm + 1, size=n)
    elif type_ == "difficult":
        firm_id = np.tile(np.arange(1, nb_firm + 1), n // nb_firm + 1)[:n]
    else:
        raise ValueError(f"Unknown type of dgp: {type_!r}")

    x1 = rng.standard_normal(n)
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


def simulate_bipartite(
    n_workers=10_000,
    n_time=5,
    firm_size=10,
    n_firm_types=5,
    n_worker_types=5,
    p_move=0.5,
    c_sort=1.0,
    c_netw=1.0,
    c_sig=1.0,
    alpha_sig=1.0,
    psi_sig=1.0,
    w_sig=1.0,
    x_sig=1.0,
    y1_beta=0.5,
    y1_sig=1.0,
    y2_beta=0.3,
    y2_sig=1.0,
    seed=None,
):
    """Simulate a bipartite labor market network.

    Generates panel data for a bipartite network of workers and firms with
    assortative matching, mobility, and AKM wage structure. Adapted from the
    bipartitepandas Python package.

    Parameters
    ----------
    n_workers : int
        Number of workers.
    n_time : int
        Panel length in time periods.
    firm_size : int
        Average firm size per period.
    n_firm_types : int
        Number of firm types.
    n_worker_types : int
        Number of worker types.
    p_move : float
        Mobility probability per period, in [0, 1].
    c_sort : float
        Assortative sorting strength.
    c_netw : float
        Network effect strength.
    c_sig : float
        Sorting/network shock volatility (must be non-negative).
    alpha_sig : float
        Worker fixed effect volatility (must be non-negative).
    psi_sig : float
        Firm fixed effect volatility (must be non-negative).
    w_sig : float
        Wage shock volatility (must be non-negative).
    x_sig : float
        Covariate volatility (must be non-negative).
    y1_beta : float
        Covariate effect on y1.
    y1_sig : float
        y1 error volatility (must be non-negative).
    y2_beta : float
        Covariate effect on y2.
    y2_sig : float
        y2 error volatility (must be non-negative).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: indiv_id, firm_id, wage, y, y2, x1, year, worker_type,
        firm_type, worker_fe, firm_fe.
    """
    rng = np.random.default_rng(seed)

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
    # Fixed effects using inverse normal CDF (R's qnorm)
    psi = norm.ppf(np.arange(1, n_firm_types + 1) / (n_firm_types + 1)) * psi_sig
    alpha = (
        norm.ppf(np.arange(1, n_worker_types + 1) / (n_worker_types + 1)) * alpha_sig
    )

    # Compute transition matrices
    G = np.zeros((n_worker_types, n_firm_types, n_firm_types))
    for l in range(n_worker_types):
        for k_from in range(n_firm_types):
            probs = norm.pdf((psi - c_netw * psi[k_from] - c_sort * alpha[l]) / c_sig)
            G[l, k_from, :] = probs / probs.sum()

    # Compute stationary distributions
    H = np.zeros((n_worker_types, n_firm_types))
    for l in range(n_worker_types):
        eigvals, eigvecs = np.linalg.eig(G[l].T)
        stationary_idx = np.argmin(np.abs(eigvals - 1))
        stationary_vec = np.real(eigvecs[:, stationary_idx])
        stationary_vec = np.abs(stationary_vec) / np.abs(stationary_vec).sum()
        H[l] = stationary_vec

    # ========================================================================
    # SIMULATE MOBILITY
    # ========================================================================
    worker_types = rng.integers(0, n_worker_types, size=n_workers)

    firm_types = np.zeros((n_workers, n_time), dtype=int)
    spell_ids = np.zeros((n_workers, n_time), dtype=int)
    spell_counter = 0

    for i in range(n_workers):
        l = worker_types[i]

        # Initial firm placement
        firm_types[i, 0] = rng.choice(n_firm_types, p=H[l])
        spell_ids[i, 0] = spell_counter
        spell_counter += 1

        # Mobility decisions for subsequent periods
        for t in range(1, n_time):
            if rng.random() < p_move:
                firm_types[i, t] = rng.choice(
                    n_firm_types, p=G[l, firm_types[i, t - 1]]
                )
                spell_ids[i, t] = spell_counter
                spell_counter += 1
            else:
                firm_types[i, t] = firm_types[i, t - 1]
                spell_ids[i, t] = spell_ids[i, t - 1]

    # ========================================================================
    # CONSTRUCT PANEL
    # ========================================================================
    indiv_id = np.repeat(np.arange(n_workers), n_time)
    time = np.tile(np.arange(n_time), n_workers)
    worker_type = np.repeat(worker_types, n_time)
    firm_type = firm_types.ravel()
    spell = spell_ids.ravel()

    # Compute spell sizes for firm ID assignment
    panel = pd.DataFrame(
        {
            "indiv_id": indiv_id,
            "year": time,
            "worker_type": worker_type,
            "firm_type": firm_type,
            "spell": spell,
        }
    )

    spell_summary = (
        panel.groupby(["spell", "firm_type"])
        .size()
        .reset_index(name="spell_size")
    )

    # Assign firm IDs
    firm_ids = np.zeros(len(spell_summary), dtype=int)
    for k_type in spell_summary["firm_type"].unique():
        k_mask = spell_summary["firm_type"] == k_type
        k_spells = spell_summary.loc[k_mask]

        total_obs = k_spells["spell_size"].sum()
        n_firms = max(1, round(total_obs / (firm_size * n_time)))

        assigned = rng.integers(0, n_firms, size=len(k_spells))
        # Map to contiguous IDs
        unique_assigned = np.unique(assigned)
        mapping = {v: idx for idx, v in enumerate(unique_assigned)}
        firm_ids[k_mask.values] = np.array([mapping[a] for a in assigned])

    spell_summary["firm_id"] = firm_ids

    # Merge firm IDs back to panel
    panel = panel.merge(spell_summary[["spell", "firm_id"]], on="spell")

    # Generate fixed effects and outcomes
    n_obs = len(panel)
    panel["worker_fe"] = alpha[panel["worker_type"].values]
    panel["firm_fe"] = psi[panel["firm_type"].values]
    panel["wage"] = panel["worker_fe"] + panel["firm_fe"] + rng.standard_normal(n_obs) * w_sig

    panel["x1"] = rng.standard_normal(n_obs) * x_sig
    panel["y"] = (
        panel["worker_fe"]
        + panel["firm_fe"]
        + y1_beta * panel["x1"]
        + rng.standard_normal(n_obs) * y1_sig
    )
    panel["y2"] = (
        panel["worker_fe"]
        + panel["firm_fe"]
        + y2_beta * panel["x1"]
        + rng.standard_normal(n_obs) * y2_sig
    )

    # Sort and return
    panel = panel.sort_values(["indiv_id", "year"]).reset_index(drop=True)

    return panel[
        [
            "indiv_id",
            "firm_id",
            "wage",
            "y",
            "y2",
            "x1",
            "year",
            "worker_type",
            "firm_type",
            "worker_fe",
            "firm_fe",
        ]
    ]
