from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["dgp", "n_fe", "n_obs", "backend"], as_index=False)["time"]
        .agg(median="median", min="min", max="max")
        .sort_values(["dgp", "n_fe", "n_obs", "backend"])
    )


def plot_benchmarks(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create a faceted benchmark plot (rows=dgp, cols=n_fe)."""
    if results_df.empty:
        return

    summary = _aggregate(results_df)
    dgps = sorted(summary["dgp"].unique())
    n_fes = sorted(summary["n_fe"].unique())
    n_obs_vals = sorted(summary["n_obs"].unique())
    backends = sorted(summary["backend"].unique())

    fig, axes = plt.subplots(
        len(dgps),
        len(n_fes),
        figsize=(5.5 * len(n_fes), 4.0 * len(dgps)),
        squeeze=False,
    )

    x = np.array(n_obs_vals)

    for row_idx, dgp in enumerate(dgps):
        for col_idx, n_fe in enumerate(n_fes):
            ax = axes[row_idx][col_idx]
            subset = summary[(summary["dgp"] == dgp) & (summary["n_fe"] == n_fe)]

            if subset.empty:
                ax.set_axis_off()
                continue

            for backend in backends:
                backend_df = (
                    subset[subset["backend"] == backend]
                    .set_index("n_obs")
                    .reindex(n_obs_vals)
                )

                medians = backend_df["median"].to_numpy(dtype=float)
                mins = backend_df["min"].to_numpy(dtype=float)
                maxs = backend_df["max"].to_numpy(dtype=float)

                valid = ~np.isnan(medians)
                if not np.any(valid):
                    continue

                ax.plot(
                    x[valid],
                    medians[valid],
                    marker="o",
                    label=backend,
                )
                ax.fill_between(
                    x[valid],
                    mins[valid],
                    maxs[valid],
                    alpha=0.15,
                )

            ax.set_title(f"{dgp} | n_fe={n_fe}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{n:,}" for n in n_obs_vals], rotation=30, ha="right")
            ax.set_xlabel("n_obs")
            ax.set_ylabel("Time (s)")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(backends)))

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
