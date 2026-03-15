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
_AKM_SWEEP_XLABELS = {
    "scale": "Observations",
    "sorting": "Sorting",
    "mobility": "Mobility",
    "size": "Firm Size Shape",
    "fragmentation": "Industry Fragmentation",
    "varratio": "Var(alpha) / Var(psi)",
    "saturation": "FE Saturation",
    "unbalanced": "Short-Panel Share",
    "interaction": "Sorting x Mobility",
}
_AKM_SWEEP_BASELINE_POINTS = {
    "sorting": (1.5, "rho=1.00"),
    "mobility": (1.5, "delta=0.20"),
    "size": (2.5, "gamma=1.00"),
    "varratio": (3.5, "2.00"),
    "unbalanced": (0.5, "0%"),
}
_AKM_SWEEP_TICK_LABELS = {
    "scale": {
        1: "10,000",
        2: "100,000",
        3: "1,000,000",
        4: "10,000,000",
    },
    "sorting": {
        1: "rho=0.00",
        2: "rho=5.00",
        3: "rho=20.00",
    },
    "mobility": {
        1: "delta=0.50",
        2: "delta=0.05",
        3: "delta=0.01",
    },
    "size": {
        1: "gamma=100.00",
        2: "gamma=2.00",
        3: "gamma=0.50",
    },
    "fragmentation": {
        1: "S=1",
        2: "S=5, lambda=0.50",
        3: "S=5, lambda=0.95",
        4: "S=20, lambda=0.95",
        5: "S=50, lambda=0.99",
    },
    "varratio": {
        1: "0.10",
        2: "0.50",
        3: "1.00",
        4: "5.00",
        5: "10.00",
    },
    "saturation": {
        1: "100k/1k/T10",
        2: "100k/10k/T10",
        3: "100k/50k/T10",
        4: "100k/90k/T10",
        5: "500k/50k/T2",
        6: "450k/400k/T2",
    },
    "unbalanced": {
        1: "10%",
        2: "25%",
        3: "50%",
        4: "75%",
    },
    "interaction": {
        1: "rho=0.00, delta=0.50",
        2: "rho=5.00, delta=0.50",
        3: "rho=0.00, delta=0.02",
        4: "rho=5.00, delta=0.02",
        5: "rho=20.00, delta=0.02",
    },
}


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
    *,
    y_label: str = "Time (s)",
    y_scale: str = "log",
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
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_yscale(y_scale)
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


def _dgp_output_path(output_path: Path, dgp: str, *, suffix: str = "") -> Path:
    stem = f"{output_path.stem}_{dgp}{suffix}"
    return output_path.with_name(stem + output_path.suffix)


def _parse_akm_sweep_dgp(dgp: str) -> tuple[str, int, str] | None:
    if dgp == "akm_baseline":
        return None

    parts = dgp.split("_")
    if len(parts) != 3 or parts[0] != "akm":
        return None

    family = parts[1]
    try:
        order = int(parts[2])
    except ValueError:
        return None

    if (
        family not in _AKM_SWEEP_TICK_LABELS
        or order not in _AKM_SWEEP_TICK_LABELS[family]
    ):
        return None

    return family, order, _AKM_SWEEP_TICK_LABELS[family][order]


def _akm_sweep_plot_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    baseline = summary[summary["dgp"] == "akm_baseline"]

    for dgp in sorted(summary["dgp"].unique()):
        parsed = _parse_akm_sweep_dgp(dgp)
        if parsed is None:
            continue
        family, x_order, x_tick_label = parsed
        subset = summary[summary["dgp"] == dgp]
        if subset.empty:
            continue
        family_rows = subset.copy()
        family_rows["family"] = family
        family_rows["x_order"] = x_order
        family_rows["x_tick_label"] = x_tick_label
        family_rows["x_label"] = _AKM_SWEEP_XLABELS[family]
        rows.extend(family_rows.to_dict("records"))

        if not baseline.empty and family in _AKM_SWEEP_BASELINE_POINTS:
            baseline_order, baseline_label = _AKM_SWEEP_BASELINE_POINTS[family]
            baseline_rows = baseline.copy()
            baseline_rows["family"] = family
            baseline_rows["x_order"] = baseline_order
            baseline_rows["x_tick_label"] = baseline_label
            baseline_rows["x_label"] = _AKM_SWEEP_XLABELS[family]
            rows.extend(baseline_rows.to_dict("records"))

    return pd.DataFrame(rows)


def _plot_sweep_figure(
    family_summary: pd.DataFrame,
    styles: dict[str, dict],
    output_path: Path,
    *,
    y_label: str,
    y_scale: str,
) -> None:
    family = family_summary["family"].iloc[0]
    x_label = family_summary["x_label"].iloc[0]
    n_fes = sorted(family_summary["n_fe"].unique())
    x_orders = sorted(family_summary["x_order"].unique())
    tick_labels = (
        family_summary[["x_order", "x_tick_label"]]
        .drop_duplicates()
        .sort_values("x_order")["x_tick_label"]
        .tolist()
    )
    backends = sorted(family_summary["backend"].unique())

    fig, axes = plt.subplots(
        1,
        len(n_fes),
        figsize=(5 * len(n_fes), 4.2),
        squeeze=False,
    )

    for col_idx, n_fe in enumerate(n_fes):
        ax = axes[0][col_idx]
        subset = family_summary[family_summary["n_fe"] == n_fe]

        for backend in backends:
            style = styles[backend]
            backend_df = (
                subset[subset["backend"] == backend]
                .sort_values("x_order")
                .drop_duplicates(subset=["x_order"], keep="last")
                .set_index("x_order")
                .reindex(x_orders)
            )
            medians = backend_df["median"].to_numpy(dtype=float)
            mins = backend_df["min"].to_numpy(dtype=float)
            maxs = backend_df["max"].to_numpy(dtype=float)
            x = np.arange(len(x_orders), dtype=float)
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

        ax.set_title(
            f"AKM Sweep: {_dgp_label(family)}  |  {n_fe} FE",
            fontsize=11,
            fontweight="semibold",
            pad=8,
        )
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_yscale(y_scale)
        ax.set_xticks(np.arange(len(x_orders), dtype=float), tick_labels)
        ax.tick_params(axis="x", rotation=30)
        _apply_common_style(ax)

    _add_legend(fig, axes, ncol=max(1, len(backends)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_akm_sweep_benchmarks(summary: pd.DataFrame, output_path: Path) -> bool:
    if "akm_sweep" not in output_path.stem:
        return False

    sweep_summary = _akm_sweep_plot_rows(summary)
    if sweep_summary.empty:
        return False

    styles = _build_styles(sorted(sweep_summary["backend"].unique()))
    for family in sorted(sweep_summary["family"].unique()):
        family_summary = sweep_summary[sweep_summary["family"] == family].copy()
        _plot_sweep_figure(
            family_summary,
            styles,
            _dgp_output_path(output_path, f"akm_sweep_{family}"),
            y_label="Time (s)",
            y_scale="log",
        )

    return True


def plot_benchmarks(results_df: pd.DataFrame, output_path: Path) -> None:
    """Create publication-ready benchmark plots, one figure per benchmark DGP."""
    if results_df.empty:
        return

    summary = _aggregate(results_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _plot_akm_sweep_benchmarks(summary, output_path):
        return

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
