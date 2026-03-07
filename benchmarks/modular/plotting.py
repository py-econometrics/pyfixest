from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style generation
# ---------------------------------------------------------------------------
_PALETTE = [
    "#E24A33",
    "#348ABD",
    "#988ED5",
    "#8EBA42",
    "#FBC15E",
    "#FFB5B8",
    "#777777",
    "#E5AE38",
]
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def _build_styles(backends: list[str]) -> dict[str, dict]:
    return {
        name: {
            "color": _PALETTE[i % len(_PALETTE)],
            "marker": _MARKERS[i % len(_MARKERS)],
            "label": name,
        }
        for i, name in enumerate(backends)
    }


def _aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["dgp", "n_fe", "n_obs", "backend"], as_index=False)["time"]
        .agg(median="median", min="min", max="max")
        .sort_values(["dgp", "n_fe", "n_obs", "backend"])
    )


def _dgp_label(dgp: str) -> str:
    return dgp.replace("_", " ").title()


def _apply_common_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def _add_legend(fig: plt.Figure, axes: np.ndarray, ncol: int) -> None:
    handles, labels = [], []
    for row in axes:
        for ax in row:
            h, lab = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, lab
                break
        if handles:
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            frameon=False,
            fontsize=10,
        )


# ---------------------------------------------------------------------------
# Line plots (scaling behaviour)
# ---------------------------------------------------------------------------


def _plot_lines(
    summary: pd.DataFrame, styles: dict[str, dict], output_path: Path
) -> None:
    dgps = sorted(summary["dgp"].unique())
    n_fes = sorted(summary["n_fe"].unique())
    n_obs_vals = sorted(summary["n_obs"].unique())
    backends = sorted(summary["backend"].unique())

    fig, axes = plt.subplots(
        len(dgps),
        len(n_fes),
        figsize=(5 * len(n_fes), 3.8 * len(dgps)),
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
                style = styles[backend]
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
                    marker=style["marker"],
                    color=style["color"],
                    label=style["label"],
                    linewidth=1.8,
                    markersize=5,
                    zorder=3,
                )
                ax.fill_between(
                    x[valid],
                    mins[valid],
                    maxs[valid],
                    color=style["color"],
                    alpha=0.12,
                )

            ax.set_title(
                f"{_dgp_label(dgp)}  |  {n_fe} FE",
                fontsize=11,
                fontweight="semibold",
                pad=8,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{n:,}" for n in n_obs_vals], rotation=30, ha="right", fontsize=9
            )
            ax.set_xlabel("Observations", fontsize=10)
            ax.set_ylabel("Time (s)", fontsize=10)
            _apply_common_style(ax)

    _add_legend(fig, axes, ncol=max(1, len(backends)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bar charts (comparison at each n_obs)
# ---------------------------------------------------------------------------


def _plot_bars(
    summary: pd.DataFrame, styles: dict[str, dict], output_path: Path
) -> None:
    dgps = sorted(summary["dgp"].unique())
    n_fes = sorted(summary["n_fe"].unique())
    n_obs_vals = sorted(summary["n_obs"].unique())
    backends = sorted(summary["backend"].unique())

    fig, axes = plt.subplots(
        len(dgps),
        len(n_fes),
        figsize=(5 * len(n_fes), 3.8 * len(dgps)),
        squeeze=False,
    )

    n_backends = len(backends)
    bar_width = 0.7 / max(n_backends, 1)

    for row_idx, dgp in enumerate(dgps):
        for col_idx, n_fe in enumerate(n_fes):
            ax = axes[row_idx][col_idx]
            subset = summary[(summary["dgp"] == dgp) & (summary["n_fe"] == n_fe)]

            if subset.empty:
                ax.set_axis_off()
                continue

            x_pos = np.arange(len(n_obs_vals))

            for bi, backend in enumerate(backends):
                style = styles[backend]
                backend_df = (
                    subset[subset["backend"] == backend]
                    .set_index("n_obs")
                    .reindex(n_obs_vals)
                )

                medians = backend_df["median"].to_numpy(dtype=float)
                mins = backend_df["min"].to_numpy(dtype=float)
                maxs = backend_df["max"].to_numpy(dtype=float)

                offset = (bi - (n_backends - 1) / 2) * bar_width
                yerr_lo = np.where(np.isnan(medians), 0, medians - mins)
                yerr_hi = np.where(np.isnan(medians), 0, maxs - medians)

                ax.bar(
                    x_pos + offset,
                    np.nan_to_num(medians),
                    width=bar_width * 0.9,
                    color=style["color"],
                    label=style["label"],
                    yerr=[yerr_lo, yerr_hi],
                    capsize=2,
                    error_kw={"linewidth": 0.8, "alpha": 0.6},
                    zorder=3,
                )

            ax.set_title(
                f"{_dgp_label(dgp)}  |  {n_fe} FE",
                fontsize=11,
                fontweight="semibold",
                pad=8,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [f"{n:,}" for n in n_obs_vals], rotation=30, ha="right", fontsize=9
            )
            ax.set_xlabel("Observations", fontsize=10)
            ax.set_ylabel("Time (s)", fontsize=10)
            ax.set_yscale("log")
            _apply_common_style(ax)

    _add_legend(fig, axes, ncol=max(1, len(backends)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_benchmarks(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create publication-ready benchmark plots (line + bar)."""
    if results_df.empty:
        return

    summary = _aggregate(results_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles(sorted(summary["backend"].unique()))
    _plot_lines(summary, styles, output_path)
    _plot_bars(summary, styles, output_path.with_name(output_path.stem + "_bars.png"))
