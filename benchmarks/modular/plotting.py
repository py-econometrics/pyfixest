from __future__ import annotations

from collections.abc import Callable
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
# Shared grid scaffolding
# ---------------------------------------------------------------------------

CellPlotFn = Callable[
    [plt.Axes, pd.DataFrame, list[int], list[str], dict[str, dict]], None
]


def _plot_dgp_figure(
    dgp_summary: pd.DataFrame,
    styles: dict[str, dict],
    output_path: Path,
    plot_cell: CellPlotFn,
) -> None:
    dgp = dgp_summary["dgp"].iloc[0]
    n_fes = sorted(dgp_summary["n_fe"].unique())
    n_obs_vals = sorted(dgp_summary["n_obs"].unique())
    backends = sorted(dgp_summary["backend"].unique())

    fig, axes = plt.subplots(
        1,
        len(n_fes),
        figsize=(5 * len(n_fes), 4.2),
        squeeze=False,
    )

    for col_idx, n_fe in enumerate(n_fes):
        ax = axes[0][col_idx]
        subset = dgp_summary[dgp_summary["n_fe"] == n_fe]

        if subset.empty:
            ax.set_axis_off()
            continue

        plot_cell(ax, subset, n_obs_vals, backends, styles)

        ax.set_title(
            f"{_dgp_label(dgp)}  |  {n_fe} FE",
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
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
# Per-cell plot functions
# ---------------------------------------------------------------------------


def _backend_stats(
    subset: pd.DataFrame, backend: str, n_obs_vals: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    backend_df = (
        subset[subset["backend"] == backend].set_index("n_obs").reindex(n_obs_vals)
    )
    return (
        backend_df["median"].to_numpy(dtype=float),
        backend_df["min"].to_numpy(dtype=float),
        backend_df["max"].to_numpy(dtype=float),
    )


def _line_cell(
    ax: plt.Axes,
    subset: pd.DataFrame,
    n_obs_vals: list[int],
    backends: list[str],
    styles: dict[str, dict],
) -> None:
    x = np.array(n_obs_vals)
    for backend in backends:
        style = styles[backend]
        medians, mins, maxs = _backend_stats(subset, backend, n_obs_vals)
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
            x[valid], mins[valid], maxs[valid], color=style["color"], alpha=0.12
        )
    ax.set_xscale("log")
    ax.set_xticks(x)


def _bar_cell(
    ax: plt.Axes,
    subset: pd.DataFrame,
    n_obs_vals: list[int],
    backends: list[str],
    styles: dict[str, dict],
) -> None:
    n_backends = len(backends)
    bar_width = 0.7 / max(n_backends, 1)
    x_pos = np.arange(len(n_obs_vals))

    for bi, backend in enumerate(backends):
        style = styles[backend]
        medians, mins, maxs = _backend_stats(subset, backend, n_obs_vals)
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
    ax.set_xticks(x_pos)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _dgp_output_path(output_path: Path, dgp: str, *, suffix: str = "") -> Path:
    stem = f"{output_path.stem}_{dgp}{suffix}"
    return output_path.with_name(stem + output_path.suffix)


def plot_benchmarks(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create publication-ready benchmark plots, one figure per benchmark DGP."""
    if results_df.empty:
        return

    summary = _aggregate(results_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dgps = sorted(summary["dgp"].unique())

    styles = _build_styles(sorted(summary["backend"].unique()))
    for dgp in dgps:
        dgp_summary = summary[summary["dgp"] == dgp].copy()
        _plot_dgp_figure(
            dgp_summary,
            styles,
            _dgp_output_path(output_path, dgp),
            _line_cell,
        )
        _plot_dgp_figure(
            dgp_summary,
            styles,
            _dgp_output_path(output_path, dgp, suffix="_bars"),
            _bar_cell,
        )
