"""
Benchmark: LSMR preconditioner on a worker-firm-year panel.

Uses a bipartite network DGP (adapted from the bipartitepandas package)
with assortative matching, worker mobility, and AKM wage structure.
Three-way FEs: worker + firm + year.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

from pyfixest.estimation.cupy.demean_cupy_ import (
    CupyFWLDemeaner,
    create_fe_sparse_matrix,
)

# ---------------------------------------------------------------------------
# Bipartite network DGP (adapted from bipartitepandas / R implementation)
# ---------------------------------------------------------------------------


def simulate_bipartite(
    n_workers: int = 10_000,
    n_time: int = 5,
    firm_size: int = 10,
    n_firm_types: int = 5,
    n_worker_types: int = 5,
    p_move: float = 0.5,
    c_sort: float = 1.0,
    c_netw: float = 1.0,
    c_sig: float = 1.0,
    alpha_sig: float = 1.0,
    psi_sig: float = 1.0,
    w_sig: float = 1.0,
    x_sig: float = 1.0,
    y1_beta: float = 0.5,
    y1_sig: float = 1.0,
    y2_beta: float = 0.3,
    y2_sig: float = 1.0,
    survival_rate: float = 1.0,
    firm_concentration: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a bipartite labor market network.

    Generates panel data for workers and firms with assortative matching,
    mobility, and an AKM wage structure.

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
        Mobility probability per period.
    c_sort : float
        Assortative sorting strength.
    c_netw : float
        Network effect strength.
    c_sig : float
        Sorting/network shock volatility.
    alpha_sig : float
        Worker fixed effect volatility.
    psi_sig : float
        Firm fixed effect volatility.
    w_sig : float
        Wage shock volatility.
    x_sig : float
        Covariate volatility.
    y1_beta : float
        Covariate effect on y1.
    y1_sig : float
        y1 error volatility.
    y2_beta : float
        Covariate effect on y2.
    y2_sig : float
        y2 error volatility.
    survival_rate : float
        Per-period survival probability.  Each worker-year observation is
        independently kept with this probability.  1.0 = balanced panel (no
        attrition).  Lower values create an unbalanced panel where worker
        tenure, firm sizes, and year counts all vary, worsening conditioning.
    firm_concentration : float >= 0
        Controls how concentrated firm sizes are.  0 = uniform assignment
        (all firms of a given type roughly equal size).  Higher values use
        Zipf-like probabilities so a few firms absorb most spells while
        many firms stay tiny.  Good range: 0-2.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Panel with columns: worker_id, firm_id, wage, y1, y2, x1,
        year, worker_type, firm_type, worker_fe, firm_fe.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Fixed effects (inverse normal CDF, matching R's qnorm)
    # ------------------------------------------------------------------
    psi = sp_norm.ppf(np.arange(1, n_firm_types + 1) / (n_firm_types + 1)) * psi_sig
    alpha = (
        sp_norm.ppf(np.arange(1, n_worker_types + 1) / (n_worker_types + 1))
        * alpha_sig
    )

    # ------------------------------------------------------------------
    # Transition matrices  G[l, k_from, k_to]
    # ------------------------------------------------------------------
    G = np.zeros((n_worker_types, n_firm_types, n_firm_types))
    for l in range(n_worker_types):
        for k_from in range(n_firm_types):
            probs = sp_norm.pdf(
                (psi - c_netw * psi[k_from] - c_sort * alpha[l]) / c_sig
            )
            G[l, k_from, :] = probs / probs.sum()

    # ------------------------------------------------------------------
    # Stationary distributions  H[l, k]
    # ------------------------------------------------------------------
    H = np.zeros((n_worker_types, n_firm_types))
    for l in range(n_worker_types):
        eigvals, eigvecs = np.linalg.eig(G[l].T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        v = np.real(eigvecs[:, idx])
        H[l] = np.abs(v) / np.abs(v).sum()

    # ------------------------------------------------------------------
    # Simulate mobility
    # ------------------------------------------------------------------
    worker_types = rng.integers(0, n_worker_types, size=n_workers)

    # firm_type_mat[i, t] = firm type of worker i at time t
    firm_type_mat = np.zeros((n_workers, n_time), dtype=np.int64)
    spell_mat = np.zeros((n_workers, n_time), dtype=np.int64)
    spell_counter = 0

    for i in range(n_workers):
        l = worker_types[i]
        # initial placement
        firm_type_mat[i, 0] = rng.choice(n_firm_types, p=H[l])
        spell_mat[i, 0] = spell_counter
        spell_counter += 1

        for t in range(1, n_time):
            if rng.random() < p_move:
                firm_type_mat[i, t] = rng.choice(
                    n_firm_types, p=G[l, firm_type_mat[i, t - 1]]
                )
                spell_mat[i, t] = spell_counter
                spell_counter += 1
            else:
                firm_type_mat[i, t] = firm_type_mat[i, t - 1]
                spell_mat[i, t] = spell_mat[i, t - 1]

    # ------------------------------------------------------------------
    # Build long-format panel
    # ------------------------------------------------------------------
    worker_id = np.repeat(np.arange(n_workers), n_time)
    year = np.tile(np.arange(n_time), n_workers)
    wtype = np.repeat(worker_types, n_time)
    ftype = firm_type_mat.ravel()
    spell = spell_mat.ravel()

    panel = pd.DataFrame(
        {
            "worker_id": worker_id,
            "year": year,
            "worker_type": wtype,
            "firm_type": ftype,
            "spell": spell,
        }
    )

    # ------------------------------------------------------------------
    # Assign firm IDs within each firm type
    # ------------------------------------------------------------------
    spell_summary = (
        panel.groupby(["spell", "firm_type"])
        .size()
        .reset_index(name="spell_size")
    )

    firm_ids = np.zeros(len(spell_summary), dtype=np.int64)
    firm_id_offset = 0

    for k_type in spell_summary["firm_type"].unique():
        mask = spell_summary["firm_type"] == k_type
        k_spells = spell_summary.loc[mask]
        total_obs = k_spells["spell_size"].sum()
        n_firms = max(1, round(total_obs / (firm_size * n_time)))

        if firm_concentration > 0 and n_firms > 1:
            ranks = np.arange(1, n_firms + 1, dtype=np.float64)
            zipf_probs = 1.0 / ranks**firm_concentration
            zipf_probs /= zipf_probs.sum()
            assigned = rng.choice(n_firms, size=len(k_spells), p=zipf_probs)
        else:
            assigned = rng.integers(0, n_firms, size=len(k_spells))
        # remap to contiguous IDs starting from firm_id_offset
        unique_assigned = np.unique(assigned)
        remap = {v: firm_id_offset + j for j, v in enumerate(unique_assigned)}
        firm_ids[mask.values] = np.array([remap[a] for a in assigned])
        firm_id_offset += len(unique_assigned)

    spell_summary["firm_id"] = firm_ids

    panel = panel.merge(
        spell_summary[["spell", "firm_id"]], on="spell", how="left"
    )

    # ------------------------------------------------------------------
    # Generate wages and outcomes
    # ------------------------------------------------------------------
    n = len(panel)
    panel["worker_fe"] = alpha[panel["worker_type"].values]
    panel["firm_fe"] = psi[panel["firm_type"].values]
    panel["wage"] = (
        panel["worker_fe"] + panel["firm_fe"] + rng.standard_normal(n) * w_sig
    )
    panel["x1"] = rng.standard_normal(n) * x_sig
    panel["y1"] = (
        panel["worker_fe"]
        + panel["firm_fe"]
        + y1_beta * panel["x1"]
        + rng.standard_normal(n) * y1_sig
    )
    panel["y2"] = (
        panel["worker_fe"]
        + panel["firm_fe"]
        + y2_beta * panel["x1"]
        + rng.standard_normal(n) * y2_sig
    )

    # ------------------------------------------------------------------
    # Attrition (unbalanced panel)
    # ------------------------------------------------------------------
    if survival_rate < 1.0:
        keep = rng.random(n) < survival_rate
        panel = panel.loc[keep].reset_index(drop=True)
        # Drop workers/firms that lost all observations
        panel = panel.groupby("worker_id").filter(lambda g: len(g) >= 1)
        panel = panel.groupby("firm_id").filter(lambda g: len(g) >= 1)
        panel = panel.reset_index(drop=True)

    panel = panel.sort_values(["worker_id", "year"]).reset_index(drop=True)

    return panel[
        [
            "worker_id",
            "firm_id",
            "wage",
            "y1",
            "y2",
            "x1",
            "year",
            "worker_type",
            "firm_type",
            "worker_fe",
            "firm_fe",
        ]
    ]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_once(demeaner, x, flist, weights, D):
    start = time.perf_counter()
    x_dm, success = demeaner.demean(x, flist, weights, fe_sparse_matrix=D)
    elapsed = time.perf_counter() - start
    resid_norm = np.linalg.norm(x_dm)
    return elapsed, success, resid_norm


def benchmark_panel(label, panel, n_repeats=3):
    """Run preconditioned vs unpreconditioned on the given panel."""
    fe_cols = ["worker_id", "firm_id", "year"]
    x_cols = ["wage", "x1"]

    x = panel[x_cols].values.astype(np.float64)
    flist = panel[fe_cols].values.astype(np.uint64)
    weights = np.ones(len(panel))

    fe_df = panel[fe_cols].astype("category")
    D = create_fe_sparse_matrix(fe_df)

    n_workers = panel["worker_id"].nunique()
    n_firms = panel["firm_id"].nunique()
    n_years = panel["year"].nunique()

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(
        f"  obs={len(panel):,}  workers={n_workers:,}  "
        f"firms={n_firms:,}  years={n_years}"
    )
    print(f"  D shape: {D.shape}  nnz: {D.nnz:,}")
    print(f"  x columns: {x_cols}  ({x.shape[1]} regressors)")
    print(f"{'=' * 72}")

    for use_precond in [False, True]:
        tag = "preconditioned" if use_precond else "   unprecond.  "
        times = []
        for _ in range(n_repeats):
            demeaner = CupyFWLDemeaner(
                use_gpu=False,
                warn_on_cpu_fallback=False,
                use_preconditioner=use_precond,
            )
            elapsed, success, resid_norm = _run_once(
                demeaner, x, flist, weights, D
            )
            times.append(elapsed)

        med = np.median(times)
        print(
            f"  [{tag}]  median={med:.4f}s  "
            f"success={success}  ||residual||={resid_norm:.6e}"
        )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def main():
    print("LSMR Preconditioner Benchmark — Worker-Firm-Year Panel")
    print("Bipartite network DGP with assortative matching")
    print("Three-way FEs: worker + firm + year")
    print("SciPy CPU-only (use_gpu=False)\n")

    # ---- Small panel, low mobility ----
    panel = simulate_bipartite(
        n_workers=10_000, n_time=10, firm_size=10,
        n_firm_types=5, n_worker_types=5,
        p_move=0.1, c_sort=0.5, seed=1,
    )
    benchmark_panel("Small, low mobility (10k workers, 10 periods)", panel)

    # ---- Medium panel, moderate mobility ----
    panel = simulate_bipartite(
        n_workers=100_000, n_time=10, firm_size=20,
        n_firm_types=10, n_worker_types=5,
        p_move=0.3, c_sort=1.0, seed=2,
    )
    benchmark_panel("Medium, moderate mobility (100k workers, 10 periods)", panel)

    # ---- Medium panel, high mobility, strong sorting ----
    panel = simulate_bipartite(
        n_workers=100_000, n_time=15, firm_size=10,
        n_firm_types=10, n_worker_types=10,
        p_move=0.5, c_sort=2.0, c_netw=2.0, seed=3,
    )
    benchmark_panel(
        "Medium, high mobility + strong sorting (100k workers, 15 periods)",
        panel,
    )

    # ---- Large panel, moderate mobility ----
    panel = simulate_bipartite(
        n_workers=500_000, n_time=10, firm_size=20,
        n_firm_types=10, n_worker_types=5,
        p_move=0.3, c_sort=1.0, seed=4,
    )
    benchmark_panel("Large, moderate mobility (500k workers, 10 periods)", panel)

    # ---- Large panel, high mobility, many firm types ----
    panel = simulate_bipartite(
        n_workers=500_000, n_time=15, firm_size=10,
        n_firm_types=20, n_worker_types=10,
        p_move=0.5, c_sort=1.5, c_netw=1.5, seed=5,
    )
    benchmark_panel(
        "Large, high mobility + many types (500k workers, 15 periods)", panel
    )

    # ---- Very large panel ----
    panel = simulate_bipartite(
        n_workers=1_000_000, n_time=10, firm_size=20,
        n_firm_types=15, n_worker_types=10,
        p_move=0.3, c_sort=1.0, seed=6,
    )
    benchmark_panel("Very large (1M workers, 10 periods)", panel)

    # ---- Pathological: Zipf firms + attrition ----
    # firm_concentration=1.5 creates Zipf-distributed firm sizes: a few
    # mega-firms with thousands of obs alongside many micro-firms with 1-2.
    # Attrition (survival_rate=0.4) adds worker-tenure variation.  Together
    # these widen the spread of column norms in D, badly conditioning D'D.
    panel = simulate_bipartite(
        n_workers=100_000, n_time=15, firm_size=5,
        n_firm_types=20, n_worker_types=5,
        p_move=0.15, c_sort=1.5, c_netw=1.0,
        survival_rate=0.4, firm_concentration=1.5, seed=7,
    )
    benchmark_panel(
        "Pathological: Zipf firms + attrition (100k workers)",
        panel,
    )


if __name__ == "__main__":
    main()
