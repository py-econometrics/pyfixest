"""Benchmark feglm acceleration: warm-start + adaptive tolerance."""

import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import pyfixest as pf


def simulate_bipartite(
    n_workers=10000,
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
    seed=42,
):
    """Simulate bipartite labor market network data."""
    rng = np.random.default_rng(seed)

    # Generate fixed effects using inverse normal CDF
    psi = norm.ppf(np.arange(1, n_firm_types + 1) / (n_firm_types + 1)) * psi_sig
    alpha = norm.ppf(np.arange(1, n_worker_types + 1) / (n_worker_types + 1)) * alpha_sig

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
        stationary_vec = np.abs(np.real(eigvecs[:, stationary_idx]))
        H[l] = stationary_vec / stationary_vec.sum()

    # Generate worker types
    worker_types = rng.integers(0, n_worker_types, size=n_workers)

    # Simulate mobility
    firm_types_mat = np.zeros((n_workers, n_time), dtype=int)
    spell_ids = np.zeros((n_workers, n_time), dtype=int)
    spell_counter = 0

    for i in range(n_workers):
        l = worker_types[i]
        firm_types_mat[i, 0] = rng.choice(n_firm_types, p=H[l])
        spell_ids[i, 0] = spell_counter
        spell_counter += 1

        for t in range(1, n_time):
            if rng.random() < p_move:
                firm_types_mat[i, t] = rng.choice(
                    n_firm_types, p=G[l, firm_types_mat[i, t - 1]]
                )
                spell_ids[i, t] = spell_counter
                spell_counter += 1
            else:
                firm_types_mat[i, t] = firm_types_mat[i, t - 1]
                spell_ids[i, t] = spell_ids[i, t - 1]

    # Construct panel
    indiv_id = np.repeat(np.arange(n_workers), n_time)
    year = np.tile(np.arange(n_time), n_workers)
    worker_type = np.repeat(worker_types, n_time)
    firm_type = firm_types_mat.flatten()
    spell = spell_ids.flatten()

    # Assign firm IDs per spell
    spell_df = pd.DataFrame({"spell": spell, "firm_type": firm_type})
    spell_summary = spell_df.groupby(["spell", "firm_type"]).size().reset_index(name="spell_size")

    firm_id_arr = np.zeros(len(spell_summary), dtype=int)
    for k_type in spell_summary["firm_type"].unique():
        mask = spell_summary["firm_type"] == k_type
        k_spells = spell_summary[mask]
        total_obs = k_spells["spell_size"].sum()
        n_firms = max(1, round(total_obs / (firm_size * n_time)))
        assigned = rng.integers(0, n_firms, size=mask.sum())
        firm_id_arr[mask.values] = assigned

    spell_summary["firm_id"] = firm_id_arr

    # Build panel DataFrame
    panel = pd.DataFrame({
        "indiv_id": indiv_id,
        "year": year,
        "worker_type": worker_type,
        "firm_type": firm_type,
        "spell": spell,
    })
    panel = panel.merge(spell_summary[["spell", "firm_id"]], on="spell", how="left")

    n = len(panel)
    panel["worker_fe"] = alpha[panel["worker_type"].values]
    panel["firm_fe"] = psi[panel["firm_type"].values]
    panel["x1"] = rng.normal(0, x_sig, n)

    # Linear predictor for binary outcome
    latent = panel["worker_fe"] + panel["firm_fe"] + y1_beta * panel["x1"]
    prob = 1 / (1 + np.exp(-latent))
    panel["Y_bin"] = (rng.random(n) < prob).astype(int)

    panel = panel.sort_values(["indiv_id", "year"]).reset_index(drop=True)

    # Convert IDs to categoricals for fixed effects
    panel["indiv_id"] = panel["indiv_id"].astype(str)
    panel["firm_id"] = panel["firm_id"].astype(str)
    panel["year"] = panel["year"].astype(str)

    return panel


if __name__ == "__main__":
    print("Generating bipartite network data...")
    data = simulate_bipartite(n_workers=50_000, n_time=5, firm_size=10)
    print(f"Panel size: {len(data):,} obs, {data['indiv_id'].nunique():,} workers, "
          f"{data['firm_id'].nunique():,} firms")
    print(f"Y_bin mean: {data['Y_bin'].mean():.3f}")
    print()

    fml = "Y_bin ~ x1 | indiv_id + firm_id + year"

    # Warm-up run
    print("Warm-up run...")
    pf.feglm(fml, data, family="logit", accelerate=True)
    print()

    # Benchmark: accelerate=True
    n_runs = 5
    print(f"Benchmark: accelerate=True ({n_runs} runs)")
    times_accel = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fit_accel = pf.feglm(fml, data, family="logit", accelerate=True)
        t1 = time.perf_counter()
        times_accel.append(t1 - t0)
        print(f"  Run {i+1}: {t1 - t0:.3f}s")

    print()

    # Benchmark: accelerate=False
    print(f"Benchmark: accelerate=False ({n_runs} runs)")
    times_no_accel = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        fit_no_accel = pf.feglm(fml, data, family="logit", accelerate=False)
        t1 = time.perf_counter()
        times_no_accel.append(t1 - t0)
        print(f"  Run {i+1}: {t1 - t0:.3f}s")

    print()
    print("=" * 50)
    mean_accel = np.mean(times_accel)
    mean_no_accel = np.mean(times_no_accel)
    print(f"Mean time (accelerate=True):  {mean_accel:.3f}s")
    print(f"Mean time (accelerate=False): {mean_no_accel:.3f}s")
    print(f"Speedup: {mean_no_accel / mean_accel:.2f}x")
    print()

    # Verify numerical equivalence
    coef_accel = fit_accel.coef()
    coef_no_accel = fit_no_accel.coef()
    max_diff = np.max(np.abs(coef_accel.values - coef_no_accel.values))
    print(f"Max coefficient difference: {max_diff:.2e}")
    print(f"Coefficients (accelerated):     {coef_accel.values}")
    print(f"Coefficients (non-accelerated): {coef_no_accel.values}")
