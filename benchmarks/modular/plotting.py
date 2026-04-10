from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
_FE_LABELS = {
    2: "Worker + Year",
    3: "Worker + Firm + Year",
    4: "Worker + Firm + Year + Occupation",
}
_AKM_SWEEP_XLABELS = {
    "scale": "Observations",
    "sorting": "Sorting",
    "mobility": "Mobility",
    "interaction": "Sorting x Mobility",
    "freeze": "Frozen Markets (of 5)",
    "occlambda": "Occupation-Firm Nesting",
    "occsize": "Occupation Dimensionality",
}
_AKM_SWEEP_BASELINE_POINTS = {
    "sorting": (1.5, "rho=1.00"),
    "mobility": (2.5, "delta=0.10"),
    "occlambda": (2.5, "occ_lambda=0.50"),
    "occsize": (2.5, "n_occ=200"),
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
        4: "rho=50.00",
        5: "rho=100.00",
    },
    "mobility": {
        1: "delta=1.00",
        2: "delta=0.50",
        3: "delta=0.05",
        4: "delta=0.01",
        5: "delta=0.005",
        6: "delta=0.001",
    },
    "interaction": {
        1: "rho=0.00, delta=0.50",
        2: "rho=20.00, delta=0.50",
        3: "rho=0.00, delta=0.02",
        4: "rho=20.00, delta=0.02",
    },
    "freeze": {
        1: "0/5",
        2: "1/5",
        3: "2/5",
        4: "3/5",
        5: "4/5",
        6: "5/5",
    },
    "occlambda": {
        1: "occ_lambda=0.01",
        2: "occ_lambda=0.20",
        3: "occ_lambda=0.90",
        4: "occ_lambda=1.00",
    },
    "occsize": {
        1: "n_occ=10",
        2: "n_occ=50",
        3: "n_occ=1000",
        4: "n_occ=5000",
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


def _filter_backends(
    results_df: pd.DataFrame, figure_backends: Iterable[str] | None
) -> pd.DataFrame:
    if figure_backends is None:
        return results_df

    selected = list(dict.fromkeys(figure_backends))
    if not selected:
        return results_df.iloc[0:0].copy()

    return results_df[results_df["backend"].isin(selected)].copy()


def _aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby(["dgp", "k", "n_fe", "n_obs", "backend"], as_index=False)[
            "time"
        ]
        .agg(median="median")
        .sort_values(["dgp", "k", "n_fe", "n_obs", "backend"])
    )


def _dgp_label(dgp: str) -> str:
    return dgp.replace("_", " ").title()


def _apply_common_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0,), numticks=20))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _ensure_min_yticks(ax: plt.Axes, min_ticks: int = 2) -> None:
    """Widen y-limits if fewer than *min_ticks* major ticks are visible."""
    ax.figure.canvas.draw()
    ticks = [
        t
        for t in ax.yaxis.get_major_locator().tick_values(*ax.get_ylim())
        if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]
    ]
    if len(ticks) >= min_ticks:
        return
    ylo, yhi = ax.get_ylim()
    log_lo, log_hi = np.log10(ylo), np.log10(yhi)
    # expand to cover at least two powers of 10
    mid = (log_lo + log_hi) / 2
    half_span = max((log_hi - log_lo) / 2, 0.6)
    ax.set_ylim(10 ** (mid - half_span), 10 ** (mid + half_span))


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
    k_vals = sorted(dgp_summary["k"].unique())
    n_fes = sorted(dgp_summary["n_fe"].unique())
    n_obs_vals = sorted(dgp_summary["n_obs"].unique())
    backends = sorted(dgp_summary["backend"].unique())

    fig, axes = plt.subplots(
        len(k_vals),
        len(n_fes),
        figsize=(5 * len(n_fes), 3.8 * len(k_vals)),
        sharey=True,
        squeeze=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(len(k_vals), len(n_fes))

    for row_idx, k in enumerate(k_vals):
        for col_idx, n_fe in enumerate(n_fes):
            ax = axes[row_idx][col_idx]
            subset = dgp_summary[
                (dgp_summary["k"] == k) & (dgp_summary["n_fe"] == n_fe)
            ]

            if subset.empty:
                ax.set_axis_off()
                continue

            plot_cell(ax, subset, n_obs_vals, backends, styles)

            ax.set_title(
                f"{_dgp_label(dgp)}  |  {_FE_LABELS.get(n_fe, f'{n_fe} FE')}  |  k={k}",
                fontsize=11,
                fontweight="semibold",
                pad=8,
            )
            ax.set_xticklabels(
                [f"{n:,}" for n in n_obs_vals], rotation=30, ha="right", fontsize=9
            )
            ax.set_xlabel("Observations", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"k={k}\n{y_label}", fontsize=10)
            ax.set_yscale(y_scale)
            _apply_common_style(ax)

    for row in axes:
        for ax in row:
            if ax.axison:
                _ensure_min_yticks(ax)
    _add_legend(fig, axes, ncol=max(1, len(backends)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-cell plot functions
# ---------------------------------------------------------------------------


def _backend_medians(
    subset: pd.DataFrame, backend: str, n_obs_vals: list[int]
) -> np.ndarray:
    backend_df = (
        subset[subset["backend"] == backend].set_index("n_obs").reindex(n_obs_vals)
    )
    return backend_df["median"].to_numpy(dtype=float)


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
        medians = _backend_medians(subset, backend, n_obs_vals)
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
    ax.set_xscale("log")
    ax.set_xticks(x)


def _dgp_output_path(output_path: Path, dgp: str) -> Path:
    stem = f"{output_path.stem}_{dgp}"
    return output_path.with_name(stem + output_path.suffix)


def _parse_akm_sweep_dgp(dgp: str) -> tuple[str, int, str] | None:
    if dgp == "akm_baseline":
        return None

    parts = dgp.split("_")
    if len(parts) < 3 or parts[0] != "akm":
        return None

    try:
        order = int(parts[-1])
    except ValueError:
        return None

    family = "_".join(parts[1:-1])

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

        ax.set_title(
            f"AKM Sweep: {_dgp_label(family)}  |  {_FE_LABELS.get(n_fe, f'{n_fe} FE')}",
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

    for row in axes:
        for ax in row:
            if ax.axison:
                _ensure_min_yticks(ax)
    _add_legend(fig, axes, ncol=max(1, len(backends)))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_akm_sweep_benchmarks(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    figure_dir: Path | None = None,
) -> bool:
    sweep_summary = _akm_sweep_plot_rows(summary)
    if sweep_summary.empty:
        return False

    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)

    styles = _build_styles(sorted(sweep_summary["backend"].unique()))
    for family in sorted(sweep_summary["family"].unique()):
        family_summary = sweep_summary[sweep_summary["family"] == family].copy()
        if figure_dir is not None:
            fig_path = figure_dir / f"bench_{family}.png"
        else:
            fig_path = _dgp_output_path(output_path, f"akm_sweep_{family}")
        _plot_sweep_figure(
            family_summary,
            styles,
            fig_path,
            y_label="Time (s)",
            y_scale="log",
        )

    return True


def plot_benchmarks(
    results_df: pd.DataFrame,
    output_path: Path,
    *,
    figure_dir: Path | None = None,
    figure_backends: Iterable[str] | None = None,
) -> None:
    """Create publication-ready benchmark plots, one figure per benchmark DGP."""
    results_df = results_df.copy()
    if "k" not in results_df.columns:
        results_df["k"] = 1
    else:
        results_df["k"] = results_df["k"].fillna(1).astype(int)

    results_df = _filter_backends(results_df, figure_backends)
    if results_df.empty:
        return

    summary = _aggregate(results_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _plot_akm_sweep_benchmarks(summary, output_path, figure_dir=figure_dir):
        return

    dgps = sorted(summary["dgp"].unique())

    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)

    styles = _build_styles(sorted(summary["backend"].unique()))
    for dgp in dgps:
        dgp_summary = summary[summary["dgp"] == dgp].copy()
        if figure_dir is not None:
            fig_path = figure_dir / f"bench_{dgp}.png"
        else:
            fig_path = _dgp_output_path(output_path, dgp)
        _plot_dgp_figure(
            dgp_summary,
            styles,
            fig_path,
            _line_cell,
        )
