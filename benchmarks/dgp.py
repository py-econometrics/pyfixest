"""
Three-way fixed effects panel data generator.

Simulates worker x firm x year panel data with tunable difficulty parameters
for benchmarking demeaning algorithms. The DGP supports:
- Pareto-distributed firm sizes
- Worker mobility with cluster structure
- Panel unbalancedness with selection
- Firm entry and exit with selection
- Worker-firm sorting
- Connected set restriction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit


@dataclass
class DGPConfig:
    """Configuration for the three-way fixed effects DGP.

    Parameters
    ----------
    n_workers : int
        Number of workers in the economy.
    n_firms : int
        Number of initial firms.
    n_years : int
        Number of time periods.
    sigma_alpha : float
        Standard deviation of worker fixed effects.
    sigma_psi : float
        Standard deviation of firm fixed effects.
    sigma_phi : float
        Standard deviation of year fixed effects.
    sigma_epsilon : float
        Standard deviation of idiosyncratic error.
    pareto_shape : float
        Shape parameter (theta) for the Pareto firm size distribution.
        Smaller values produce more skewed distributions.
    min_firm_size : int
        Minimum firm size (s_min) for the Pareto distribution.
    p_move : float
        Probability that a worker moves firms in any given year.
    n_clusters : int
        Number of firm clusters (K). Controls labor market segmentation.
    p_between_cluster : float
        Probability that a move goes to a firm in a different cluster.
        1.0 means no cluster structure (all moves are unrestricted).
    p_observe : float
        Average probability of observing a worker-year (p_bar_obs).
        1.0 means fully balanced panel.
    selection_worker : float
        Selection parameter (delta) for worker participation.
        Positive values mean high-alpha workers are more likely observed.
    spell_concentration : float
        Spell concentration parameter (kappa). Higher values produce
        longer continuous spells of presence/absence. 1.0 = iid.
    p_survive : float
        Average annual firm survival probability (p_bar_surv).
        1.0 means no firm exit.
    selection_firm : float
        Selection parameter (gamma) for firm survival.
        Positive values mean high-psi firms are more likely to survive.
    firm_entry_rate : float
        Rate of new firm entry per year (lambda_entry), as a fraction
        of the initial firm count.
    sorting_wf : float
        Worker-firm sorting parameter (rho_wf). 0.0 = random assignment,
        1.0 = perfect positive assortative matching.
    seed : int or None
        Random seed for reproducibility.
    """

    # Scale
    n_workers: int = 10_000
    n_firms: int = 1_000
    n_years: int = 10

    # Fixed effect distributions
    sigma_alpha: float = 1.0
    sigma_psi: float = 1.0
    sigma_phi: float = 0.5
    sigma_epsilon: float = 1.0

    # Firm size distribution
    pareto_shape: float = 2.0
    min_firm_size: int = 5

    # Worker mobility
    p_move: float = 0.05
    n_clusters: int = 1
    p_between_cluster: float = 1.0

    # Panel unbalancedness
    p_observe: float = 1.0
    selection_worker: float = 0.0
    spell_concentration: float = 1.0

    # Firm entry and exit
    p_survive: float = 1.0
    selection_firm: float = 0.0
    firm_entry_rate: float = 0.0

    # Sorting
    sorting_wf: float = 0.0

    # Random seed
    seed: int | None = None


@dataclass
class DGPResult:
    """Result of a DGP simulation.

    Attributes
    ----------
    data : pd.DataFrame
        Panel data with columns: worker_id, firm_id, year, y.
        IDs are contiguous integers starting from 0 within the connected set.
    true_alpha : np.ndarray
        Worker fixed effects (length = number of observed workers).
    true_psi : np.ndarray
        Firm fixed effects (length = total firms including entrants).
    true_phi : np.ndarray
        Year fixed effects (length = n_years).
    config : DGPConfig
        The configuration used to generate the data.
    n_obs : int
        Number of observations in the final panel.
    n_workers_observed : int
        Number of workers with at least one observation.
    n_firms_observed : int
        Number of firms with at least one observation.
    connected_set_fraction : float
        Fraction of observations in the largest connected component.
    """

    data: pd.DataFrame
    true_alpha: np.ndarray
    true_psi: np.ndarray
    true_phi: np.ndarray
    config: DGPConfig
    n_obs: int
    n_workers_observed: int
    n_firms_observed: int
    connected_set_fraction: float

    def describe(self) -> str:
        """Print a summary of the generated data.

        Returns
        -------
        str
            A multi-line summary string.
        """
        df = self.data
        obs_per_worker = df.groupby("worker_id").size()
        obs_per_firm = df.groupby("firm_id").size()
        n_years = df["year"].nunique()

        # Fraction of movers: workers observed at more than one firm
        firms_per_worker = df.groupby("worker_id")["firm_id"].nunique()
        n_movers = (firms_per_worker > 1).sum()
        frac_movers = n_movers / self.n_workers_observed if self.n_workers_observed > 0 else 0.0

        lines = [
            "=== DGP Summary ===",
            f"Observations:          {self.n_obs:,}",
            f"Workers (observed):    {self.n_workers_observed:,}",
            f"Firms (observed):      {self.n_firms_observed:,}",
            f"Years:                 {n_years}",
            f"Obs/worker:            mean={obs_per_worker.mean():.1f}, "
            f"min={obs_per_worker.min()}, max={obs_per_worker.max()}",
            f"Obs/firm:              mean={obs_per_firm.mean():.1f}, "
            f"min={obs_per_firm.min()}, max={obs_per_firm.max()}",
            f"Fraction of movers:    {frac_movers:.3f}",
            f"Connected set frac:    {self.connected_set_fraction:.4f}",
        ]
        summary = "\n".join(lines)
        print(summary)
        return summary


class ThreeWayFEData:
    """Three-way fixed effects panel data generator.

    Generates synthetic panel data y_it = alpha_i + psi_J(i,t) + phi_t + eps_it
    where alpha are worker FEs, psi are firm FEs, phi are year FEs.

    Parameters
    ----------
    config : DGPConfig
        Configuration specifying all DGP parameters.
    """

    def __init__(self, config: DGPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def simulate(self) -> DGPResult:
        """Run the full DGP simulation.

        Returns
        -------
        DGPResult
            The generated panel data and metadata.
        """
        cfg = self.config

        # Step 1: Draw fixed effects
        alpha, psi, phi = self._draw_fixed_effects()

        # Step 2: Assign firm sizes and cluster memberships
        firm_sizes, firm_clusters = self._assign_firm_sizes()

        # Step 3: Initial assignment of workers to firms
        worker_firms_t0 = self._initial_assignment(firm_sizes)

        # Step 4: Apply sorting
        if cfg.sorting_wf != 0.0:
            worker_firms_t0 = self._apply_sorting(
                worker_firms_t0, alpha, psi, firm_sizes
            )

        # Step 5: Simulate mobility across years
        # worker_firm_history: shape (n_workers, n_years), entry = firm_id
        worker_firm_history, psi, firm_clusters = self._simulate_mobility(
            worker_firms_t0, psi, firm_sizes, firm_clusters
        )

        # Step 7: Simulate participation (panel unbalancedness)
        observed = self._simulate_participation(alpha)

        # Step 8: Construct panel
        df, total_obs_before_connected = self._construct_panel(
            worker_firm_history, observed, alpha, psi, phi
        )

        if len(df) == 0:
            return DGPResult(
                data=df,
                true_alpha=alpha,
                true_psi=psi,
                true_phi=phi,
                config=cfg,
                n_obs=0,
                n_workers_observed=0,
                n_firms_observed=0,
                connected_set_fraction=0.0,
            )

        # Step 9: Restrict to connected set
        df, connected_frac = self._restrict_connected_set(df, total_obs_before_connected)

        # Re-index IDs to be contiguous and re-index true effects to match
        old_workers = sorted(df["worker_id"].unique())
        old_firms = sorted(df["firm_id"].unique())
        worker_map = {old: new for new, old in enumerate(old_workers)}
        firm_map = {old: new for new, old in enumerate(old_firms)}
        df["worker_id"] = df["worker_id"].map(worker_map)
        df["firm_id"] = df["firm_id"].map(firm_map)

        n_workers_obs = len(old_workers)
        n_firms_obs = len(old_firms)

        # Re-index fixed effect arrays so index i matches new worker/firm ID i
        alpha_reindexed = alpha[np.array(old_workers)]
        psi_reindexed = psi[np.array(old_firms)]

        return DGPResult(
            data=df.reset_index(drop=True),
            true_alpha=alpha_reindexed,
            true_psi=psi_reindexed,
            true_phi=phi,
            config=cfg,
            n_obs=len(df),
            n_workers_observed=n_workers_obs,
            n_firms_observed=n_firms_obs,
            connected_set_fraction=connected_frac,
        )

    def _draw_fixed_effects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 1: Draw worker, firm, and year fixed effects.

        Returns
        -------
        alpha : np.ndarray of shape (n_workers,)
        psi : np.ndarray of shape (n_firms,)
        phi : np.ndarray of shape (n_years,)
        """
        cfg = self.config
        alpha = self.rng.normal(0, cfg.sigma_alpha, size=cfg.n_workers)
        psi = self.rng.normal(0, cfg.sigma_psi, size=cfg.n_firms)
        phi = self.rng.normal(0, cfg.sigma_phi, size=cfg.n_years)
        return alpha, psi, phi

    def _assign_firm_sizes(self) -> tuple[np.ndarray, np.ndarray]:
        """Step 2: Draw firm sizes from Pareto and assign cluster memberships.

        Firm size: s_j = floor(s_min * U^(-1/theta)) where U ~ Uniform(0,1).
        Clusters are assigned round-robin to firms sorted by size.

        Returns
        -------
        firm_sizes : np.ndarray of shape (n_firms,), dtype int
        firm_clusters : np.ndarray of shape (n_firms,), dtype int
        """
        cfg = self.config
        u = self.rng.uniform(0, 1, size=cfg.n_firms)
        firm_sizes = np.floor(
            cfg.min_firm_size * u ** (-1.0 / cfg.pareto_shape)
        ).astype(np.int64)

        # Assign clusters: sort firms by size, round-robin assignment
        if cfg.n_clusters <= 1:
            firm_clusters = np.zeros(cfg.n_firms, dtype=np.int64)
        else:
            size_order = np.argsort(firm_sizes)
            firm_clusters = np.empty(cfg.n_firms, dtype=np.int64)
            firm_clusters[size_order] = np.arange(cfg.n_firms) % cfg.n_clusters

        return firm_sizes, firm_clusters

    def _initial_assignment(self, firm_sizes: np.ndarray) -> np.ndarray:
        """Step 3: Assign workers to firms proportional to firm sizes.

        Parameters
        ----------
        firm_sizes : np.ndarray
            Size of each firm.

        Returns
        -------
        worker_firms : np.ndarray of shape (n_workers,)
            Firm assignment for each worker at t=0.
        """
        cfg = self.config
        probs = firm_sizes.astype(np.float64)
        probs /= probs.sum()
        worker_firms = self.rng.choice(cfg.n_firms, size=cfg.n_workers, p=probs)
        return worker_firms

    def _apply_sorting(
        self,
        worker_firms: np.ndarray,
        alpha: np.ndarray,
        psi: np.ndarray,
        firm_sizes: np.ndarray,
    ) -> np.ndarray:
        """Step 4: Apply worker-firm sorting via rank-based procedure.

        Workers receive a noisy rank that mixes alpha rank with a random rank.
        Firm slots are ranked by psi. Workers are matched to slots by rank.

        Parameters
        ----------
        worker_firms : np.ndarray
            Current firm assignments (will be replaced).
        alpha : np.ndarray
            Worker fixed effects.
        psi : np.ndarray
            Firm fixed effects.
        firm_sizes : np.ndarray
            Firm sizes.

        Returns
        -------
        np.ndarray
            Sorted firm assignments.
        """
        cfg = self.config
        n_workers = cfg.n_workers
        rho = cfg.sorting_wf

        # Worker side: create noisy rank
        alpha_ranks = np.argsort(np.argsort(alpha)).astype(np.float64)
        random_ranks = self.rng.uniform(0, n_workers, size=n_workers)
        noisy_ranks = rho * alpha_ranks + (1.0 - rho) * random_ranks

        # Firm side: create slot array with firm_id for each slot, sorted by psi
        # Each firm j contributes firm_sizes[j] slots
        total_slots = firm_sizes.sum()
        slot_firm_ids = np.repeat(np.arange(cfg.n_firms), firm_sizes)
        slot_psi = psi[slot_firm_ids]
        # Sort slots by psi (descending so highest psi gets highest rank)
        slot_order = np.argsort(slot_psi)

        # We need exactly n_workers slots. Sample or truncate.
        if total_slots >= n_workers:
            # Sample n_workers slots proportional to firm size (already done via repeat)
            # Use the top n_workers slots or sample uniformly
            slot_indices = self.rng.choice(total_slots, size=n_workers, replace=False)
            slot_indices.sort()
            selected_firms = slot_firm_ids[slot_order[slot_indices]]
        else:
            # More workers than slots: sample with replacement
            slot_indices = self.rng.choice(total_slots, size=n_workers, replace=True)
            slot_indices.sort()
            selected_firms = slot_firm_ids[slot_order[slot_indices]]

        # Match workers to slots by rank
        worker_order = np.argsort(noisy_ranks)
        new_firms = np.empty(n_workers, dtype=np.int64)
        new_firms[worker_order] = selected_firms

        return new_firms

    def _simulate_mobility(
        self,
        worker_firms_t0: np.ndarray,
        psi: np.ndarray,
        firm_sizes: np.ndarray,
        firm_clusters: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 5 & 6: Simulate worker mobility and firm dynamics over time.

        For each year t > 0:
          - Firms may exit (Step 6). Workers at exiting firms must move.
          - New firms may enter.
          - Each worker moves with probability p_move (or is forced if firm died).
          - Destination firm sampled proportional to firm sizes, respecting clusters.

        Parameters
        ----------
        worker_firms_t0 : np.ndarray of shape (n_workers,)
            Initial firm assignments.
        psi : np.ndarray
            Firm fixed effects (may grow with entrants).
        firm_sizes : np.ndarray
            Firm sizes (may grow with entrants).
        firm_clusters : np.ndarray
            Firm cluster memberships (may grow with entrants).

        Returns
        -------
        worker_firm_history : np.ndarray of shape (n_workers, n_years)
        psi : np.ndarray (possibly extended with entrant firms)
        firm_clusters : np.ndarray (possibly extended)
        """
        cfg = self.config
        n_workers = cfg.n_workers
        n_years = cfg.n_years

        history = np.empty((n_workers, n_years), dtype=np.int64)
        history[:, 0] = worker_firms_t0

        # Track which firms are alive
        n_firms_current = cfg.n_firms
        firm_alive = np.ones(n_firms_current, dtype=bool)

        # Mutable copies
        psi_list = list(psi)
        sizes_list = list(firm_sizes)
        clusters_list = list(firm_clusters)

        for t in range(1, n_years):
            # --- Step 6: Firm survival ---
            if cfg.p_survive < 1.0:
                firm_alive, psi_list, sizes_list, clusters_list, n_firms_current = (
                    self._firm_dynamics_step(
                        firm_alive, psi_list, sizes_list, clusters_list,
                        n_firms_current, t
                    )
                )

            # --- Step 6b: Firm entry ---
            if cfg.firm_entry_rate > 0.0:
                n_entrants = self.rng.poisson(cfg.firm_entry_rate * cfg.n_firms)
                for _ in range(n_entrants):
                    new_psi = self.rng.normal(0, cfg.sigma_psi)
                    u = self.rng.uniform()
                    new_size = int(np.floor(cfg.min_firm_size * u ** (-1.0 / cfg.pareto_shape)))
                    new_cluster = self.rng.integers(0, max(cfg.n_clusters, 1))
                    psi_list.append(new_psi)
                    sizes_list.append(new_size)
                    clusters_list.append(new_cluster)
                    firm_alive = np.append(firm_alive, True)
                    n_firms_current = len(psi_list)

            # --- Step 5: Worker mobility ---
            prev_firms = history[:, t - 1]
            new_firms = prev_firms.copy()

            # Determine which workers need to move
            # Forced movers: workers whose firm died
            forced_movers = ~firm_alive[prev_firms]
            # Voluntary movers
            voluntary = self.rng.uniform(size=n_workers) < cfg.p_move
            voluntary &= ~forced_movers  # don't double-count
            movers = forced_movers | voluntary

            n_movers = movers.sum()
            if n_movers > 0:
                new_firms[movers] = self._draw_destinations(
                    prev_firms[movers],
                    firm_alive,
                    np.array(sizes_list, dtype=np.float64),
                    np.array(clusters_list, dtype=np.int64),
                    forced=forced_movers[movers],
                )

            history[:, t] = new_firms

        psi_out = np.array(psi_list)
        clusters_out = np.array(clusters_list)
        return history, psi_out, clusters_out

    def _firm_dynamics_step(
        self,
        firm_alive: np.ndarray,
        psi_list: list[float],
        sizes_list: list[int],
        clusters_list: list[int],
        n_firms_current: int,
        t: int,
    ) -> tuple[np.ndarray, list, list, list, int]:
        """Apply firm survival for one time step.

        Uses logistic-logit parameterization:
            p_surv(j) = expit(logit(p_bar_surv) + gamma * psi_j)

        Returns updated firm_alive and lists.
        """
        cfg = self.config
        psi_arr = np.array(psi_list)

        if cfg.p_survive <= 0.0:
            # All firms die
            firm_alive[:] = False
        elif cfg.p_survive >= 1.0:
            pass  # No exits
        else:
            # Compute per-firm survival probability
            base_logit = logit(cfg.p_survive)
            p_surv = expit(base_logit + cfg.selection_firm * psi_arr)
            # Only currently alive firms can die
            alive_mask = firm_alive.copy()
            draws = self.rng.uniform(size=len(psi_arr))
            dies = alive_mask & (draws >= p_surv)
            firm_alive[dies] = False

        return firm_alive, psi_list, sizes_list, clusters_list, n_firms_current

    def _draw_destinations(
        self,
        current_firms: np.ndarray,
        firm_alive: np.ndarray,
        firm_sizes: np.ndarray,
        firm_clusters: np.ndarray,
        forced: np.ndarray,
    ) -> np.ndarray:
        """Draw destination firms for moving workers.

        Respects cluster structure: with probability p_between_cluster, the
        worker moves to any alive firm (proportional to size). Otherwise,
        moves within the same cluster.

        Parameters
        ----------
        current_firms : np.ndarray of shape (n_movers,)
            Current firm of each mover.
        firm_alive : np.ndarray of shape (n_total_firms,), bool
            Which firms are currently alive.
        firm_sizes : np.ndarray of shape (n_total_firms,)
            Firm sizes.
        firm_clusters : np.ndarray of shape (n_total_firms,)
            Cluster memberships.
        forced : np.ndarray of shape (n_movers,), bool
            Whether each mover is forced (firm died) vs voluntary.

        Returns
        -------
        np.ndarray of shape (n_movers,)
            Destination firm for each mover.
        """
        cfg = self.config
        n_movers = len(current_firms)
        destinations = np.empty(n_movers, dtype=np.int64)

        alive_indices = np.where(firm_alive)[0]
        if len(alive_indices) == 0:
            # Edge case: no firms alive, assign to firm 0
            destinations[:] = 0
            return destinations

        alive_sizes = firm_sizes[alive_indices]
        alive_clusters = firm_clusters[alive_indices]

        # Decide between vs within cluster for each mover
        between = self.rng.uniform(size=n_movers) < cfg.p_between_cluster

        # For between-cluster moves (or if only 1 cluster): sample from all alive firms
        # For within-cluster moves: sample from same-cluster alive firms

        # Precompute cluster-level sampling weights
        if cfg.n_clusters > 1:
            cluster_masks = {}
            cluster_probs = {}
            for k in range(cfg.n_clusters):
                mask = alive_clusters == k
                if mask.any():
                    w = alive_sizes[mask].copy()
                    w /= w.sum()
                    cluster_masks[k] = np.where(mask)[0]
                    cluster_probs[k] = w

        # Global probs
        global_probs = alive_sizes.copy()
        global_probs /= global_probs.sum()

        for i in range(n_movers):
            if between[i] or cfg.n_clusters <= 1:
                # Sample from all alive firms, excluding current if voluntary
                if not forced[i]:
                    # Try to exclude current firm
                    cur = current_firms[i]
                    cur_in_alive = np.searchsorted(alive_indices, cur)
                    if cur_in_alive < len(alive_indices) and alive_indices[cur_in_alive] == cur:
                        if len(alive_indices) > 1:
                            probs = global_probs.copy()
                            probs[cur_in_alive] = 0.0
                            probs /= probs.sum()
                            idx = self.rng.choice(len(alive_indices), p=probs)
                        else:
                            idx = 0
                    else:
                        idx = self.rng.choice(len(alive_indices), p=global_probs)
                else:
                    idx = self.rng.choice(len(alive_indices), p=global_probs)
                destinations[i] = alive_indices[idx]
            else:
                # Within-cluster move
                worker_cluster = firm_clusters[current_firms[i]]
                if worker_cluster in cluster_masks:
                    c_indices = cluster_masks[worker_cluster]
                    c_probs = cluster_probs[worker_cluster]

                    if not forced[i] and len(c_indices) > 1:
                        # Exclude current firm
                        cur = current_firms[i]
                        cur_pos = np.searchsorted(alive_indices[c_indices], cur)
                        local_alive = alive_indices[c_indices]
                        if cur_pos < len(local_alive) and local_alive[cur_pos] == cur:
                            p = c_probs.copy()
                            p[cur_pos] = 0.0
                            p /= p.sum()
                            idx = self.rng.choice(len(c_indices), p=p)
                        else:
                            idx = self.rng.choice(len(c_indices), p=c_probs)
                    else:
                        idx = self.rng.choice(len(c_indices), p=c_probs)
                    destinations[i] = alive_indices[c_indices[idx]]
                else:
                    # Cluster has no alive firms, fall back to global
                    idx = self.rng.choice(len(alive_indices), p=global_probs)
                    destinations[i] = alive_indices[idx]

        return destinations

    def _simulate_participation(self, alpha: np.ndarray) -> np.ndarray:
        """Step 7: Simulate panel participation (observability).

        Uses an AR(1) latent process:
            z_it = rho_z * z_{i,t-1} + (1-rho_z) * nu_it
        where rho_z = 1 - 1/kappa.
        Worker is observed if z_it < threshold, where threshold is set
        so that the marginal probability equals:
            p_obs(i) = expit(logit(p_bar_obs) + delta * alpha_i)

        Parameters
        ----------
        alpha : np.ndarray of shape (n_workers,)
            Worker fixed effects (used for selection).

        Returns
        -------
        observed : np.ndarray of shape (n_workers, n_years), bool
            Whether each worker-year is observed.
        """
        cfg = self.config
        n_workers = cfg.n_workers
        n_years = cfg.n_years

        if cfg.p_observe >= 1.0:
            return np.ones((n_workers, n_years), dtype=bool)

        # Per-worker observation probabilities
        if cfg.selection_worker == 0.0:
            p_obs = np.full(n_workers, cfg.p_observe)
        else:
            if cfg.p_observe <= 0.0:
                return np.zeros((n_workers, n_years), dtype=bool)
            base_logit_val = logit(cfg.p_observe)
            p_obs = expit(base_logit_val + cfg.selection_worker * alpha)

        if cfg.spell_concentration <= 1.0:
            # IID participation
            u = self.rng.uniform(size=(n_workers, n_years))
            return u < p_obs[:, np.newaxis]

        # AR(1) latent process
        rho_z = 1.0 - 1.0 / cfg.spell_concentration

        # We use a latent Gaussian AR(1) and threshold it.
        # The stationary variance of z is sigma^2 = (1 - rho_z^2)^{-1} * sigma_nu^2
        # We want the marginal distribution of z to be N(0,1) so we set sigma_nu appropriately.
        sigma_nu = np.sqrt(1.0 - rho_z ** 2) if abs(rho_z) < 1.0 else 0.01

        # Draw latent process
        z = np.empty((n_workers, n_years))
        z[:, 0] = self.rng.normal(0, 1.0, size=n_workers)
        for t in range(1, n_years):
            nu = self.rng.normal(0, sigma_nu, size=n_workers)
            z[:, t] = rho_z * z[:, t - 1] + nu

        # Convert to uniform via normal CDF, then threshold at p_obs
        from scipy.stats import norm
        u = norm.cdf(z)
        observed = u < p_obs[:, np.newaxis]

        return observed

    def _construct_panel(
        self,
        worker_firm_history: np.ndarray,
        observed: np.ndarray,
        alpha: np.ndarray,
        psi: np.ndarray,
        phi: np.ndarray,
    ) -> tuple[pd.DataFrame, int]:
        """Step 8: Construct the panel DataFrame.

        Combines fixed effects and noise to produce y = alpha_i + psi_J(i,t) + phi_t + eps.

        Parameters
        ----------
        worker_firm_history : np.ndarray of shape (n_workers, n_years)
        observed : np.ndarray of shape (n_workers, n_years), bool
        alpha, psi, phi : np.ndarray
            Fixed effects.

        Returns
        -------
        df : pd.DataFrame
            Panel data with columns worker_id, firm_id, year, y.
        total_obs : int
            Total observations before connected set restriction.
        """
        cfg = self.config

        # Vectorized construction
        worker_idx, year_idx = np.where(observed)
        firm_idx = worker_firm_history[worker_idx, year_idx]

        n_obs = len(worker_idx)
        if n_obs == 0:
            return pd.DataFrame(columns=["worker_id", "firm_id", "year", "y"]), 0

        eps = self.rng.normal(0, cfg.sigma_epsilon, size=n_obs)
        y = alpha[worker_idx] + psi[firm_idx] + phi[year_idx] + eps

        df = pd.DataFrame({
            "worker_id": worker_idx,
            "firm_id": firm_idx,
            "year": year_idx,
            "y": y,
        })

        return df, n_obs

    def _restrict_connected_set(
        self, df: pd.DataFrame, total_obs: int
    ) -> tuple[pd.DataFrame, float]:
        """Step 9: Restrict to the largest connected component.

        Uses union-find on the bipartite worker-firm graph.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        total_obs : int
            Total observations before restriction.

        Returns
        -------
        df : pd.DataFrame
            Panel restricted to the largest connected component.
        connected_fraction : float
            Fraction of original observations retained.
        """
        if len(df) == 0:
            return df, 0.0

        workers = df["worker_id"].values
        firms = df["firm_id"].values

        unique_workers = np.unique(workers)
        unique_firms = np.unique(firms)
        n_w = len(unique_workers)
        n_f = len(unique_firms)

        # Map to contiguous indices for union-find
        w_map = {w: i for i, w in enumerate(unique_workers)}
        f_map = {f: i + n_w for i, f in enumerate(unique_firms)}

        # Union-Find
        n_nodes = n_w + n_f
        parent = np.arange(n_nodes, dtype=np.int64)
        rank = np.zeros(n_nodes, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        # Build edges from unique worker-firm pairs
        pairs = df[["worker_id", "firm_id"]].drop_duplicates()
        for _, row in pairs.iterrows():
            union(w_map[row["worker_id"]], f_map[row["firm_id"]])

        # Find component of each observation's worker
        worker_nodes = np.array([w_map[w] for w in workers])
        # Vectorized find using iterative path compression on the array
        components = np.array([find(wn) for wn in worker_nodes])

        # Find largest component
        comp_ids, comp_counts = np.unique(components, return_counts=True)
        largest_comp = comp_ids[np.argmax(comp_counts)]

        mask = components == largest_comp
        connected_fraction = mask.sum() / total_obs if total_obs > 0 else 1.0

        return df[mask].copy(), connected_fraction
