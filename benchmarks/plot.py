"""
Plotting and table utilities for benchmark results.

Produces bar charts, line plots, scatter plots, and formatted tables
summarizing demeaning benchmark performance across scenarios, backends,
and parameter sweeps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def format_summary_table(
    summary_df: pd.DataFrame,
    tablefmt: str = "github",
) -> str:
    """Format benchmark summary as a nice table string.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from summarize_results(), with columns:
        scenario, backend, n_obs, demean_time_median, etc.
    tablefmt : str
        Table format for tabulate (e.g. "github", "grid", "pipe", "latex").

    Returns
    -------
    str
        Formatted table string.
    """
    display = summary_df.copy()

    # Preserve original scenario order
    scenario_order = display["scenario"].unique()
    display["scenario"] = pd.Categorical(
        display["scenario"], categories=scenario_order, ordered=True,
    )
    display = display.sort_values("scenario")

    # Format columns for readability
    display["n_obs"] = display["n_obs"].apply(lambda x: f"{x:,.0f}")
    display["n_workers"] = display["n_workers"].apply(lambda x: f"{x:,.0f}")
    display["n_firms"] = display["n_firms"].apply(lambda x: f"{x:,.0f}")
    display["connected_set_fraction"] = display["connected_set_fraction"].apply(
        lambda x: f"{x:.4f}"
    )
    display["demean_time_median"] = display["demean_time_median"].apply(
        lambda x: f"{x:.3f}s"
    )
    if "demean_time_min" in display.columns:
        display["demean_time_min"] = display["demean_time_min"].apply(
            lambda x: f"{x:.3f}s"
        )
        display["demean_time_max"] = display["demean_time_max"].apply(
            lambda x: f"{x:.3f}s"
        )

    # Select and rename columns
    cols = {
        "scenario": "Scenario",
        "backend": "Backend",
        "n_obs": "Obs",
        "n_workers": "Workers",
        "n_firms": "Firms",
        "connected_set_fraction": "Conn. Set",
        "demean_time_median": "Time (med)",
        "demean_time_min": "Time (min)",
        "demean_time_max": "Time (max)",
        "demean_converged": "Converged",
        "n_features": "Cols",
    }

    # Add iteration columns when available
    if (
        "demean_n_iterations_median" in display.columns
        and display["demean_n_iterations_median"].notna().any()
    ):
        display["demean_n_iterations_median"] = display["demean_n_iterations_median"].apply(
            lambda x: f"{x:.0f}" if pd.notna(x) else "-"
        )
        cols["demean_n_iterations_median"] = "Iters (med)"

    if (
        "demean_time_per_iter_median" in display.columns
        and display["demean_time_per_iter_median"].notna().any()
    ):
        display["demean_time_per_iter_median"] = display["demean_time_per_iter_median"].apply(
            lambda x: f"{x:.6f}s" if pd.notna(x) else "-"
        )
        cols["demean_time_per_iter_median"] = "Time/Iter (med)"

    available = {k: v for k, v in cols.items() if k in display.columns}
    display = display[list(available.keys())].rename(columns=available)

    return tabulate(display, headers="keys", tablefmt=tablefmt, showindex=False)


def format_comparison_table(
    summary_df: pd.DataFrame,
    tablefmt: str = "github",
) -> str:
    """Format a backend comparison pivot table (scenarios as rows, backends as columns).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from summarize_results().
    tablefmt : str
        Table format for tabulate.

    Returns
    -------
    str
        Formatted pivot table string.
    """
    if "backend" not in summary_df.columns:
        return format_summary_table(summary_df, tablefmt)

    backends = sorted(summary_df["backend"].unique())
    if len(backends) <= 1:
        return format_summary_table(summary_df, tablefmt)

    # Preserve original scenario order
    scenario_order = summary_df["scenario"].unique()

    # Pivot: scenario x backend -> median time
    pivot = summary_df.pivot_table(
        index="scenario",
        columns="backend",
        values="demean_time_median",
        aggfunc="first",
    )
    pivot = pivot.reindex(scenario_order)

    # Add obs count
    obs_by_scenario = (
        summary_df.groupby("scenario")["n_obs"]
        .first()
        .apply(lambda x: f"{x:,.0f}")
    )
    pivot.insert(0, "Obs", obs_by_scenario)

    # Add iteration data if available
    has_iters = (
        "demean_n_iterations_median" in summary_df.columns
        and summary_df["demean_n_iterations_median"].notna().any()
    )
    if has_iters:
        iters_pivot = summary_df.pivot_table(
            index="scenario",
            columns="backend",
            values="demean_n_iterations_median",
            aggfunc="first",
        )
        for col in backends:
            if col in iters_pivot.columns:
                pivot[f"{col} iters"] = iters_pivot[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else "-"
                )

    # Format times
    for col in backends:
        if col in pivot.columns:
            pivot[col] = pivot[col].apply(
                lambda x: f"{x:.3f}s" if pd.notna(x) else "-"
            )

    # Add speedup column (fastest / others)
    time_cols = [c for c in backends if c in summary_df["backend"].unique()]
    if len(time_cols) >= 2:
        time_pivot = summary_df.pivot_table(
            index="scenario",
            columns="backend",
            values="demean_time_median",
            aggfunc="first",
        )
        fastest = time_pivot[time_cols].min(axis=1)
        for col in time_cols:
            ratio = time_pivot[col] / fastest
            pivot[f"{col} (x)"] = ratio.apply(
                lambda x: f"{x:.1f}x" if pd.notna(x) else "-"
            )

    pivot = pivot.reset_index().rename(columns={"scenario": "Scenario"})
    return tabulate(pivot, headers="keys", tablefmt=tablefmt, showindex=False)


def save_table(
    summary_df: pd.DataFrame,
    output_path: Path,
    tablefmt: str = "github",
) -> None:
    """Save formatted summary and comparison tables to a text file.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from summarize_results().
    output_path : Path
        Where to save the .txt file.
    tablefmt : str
        Table format for tabulate.
    """
    parts = []
    parts.append("=" * 70)
    parts.append("BENCHMARK RESULTS")
    parts.append("=" * 70)
    parts.append("")
    parts.append("--- Full Summary ---")
    parts.append(format_summary_table(summary_df, tablefmt))
    parts.append("")

    if "backend" in summary_df.columns and summary_df["backend"].nunique() > 1:
        parts.append("--- Backend Comparison ---")
        parts.append(format_comparison_table(summary_df, tablefmt))
        parts.append("")

    text = "\n".join(parts)
    output_path.write_text(text)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scenario_times(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str = "Demean Wall-Clock Time by Scenario",
) -> None:
    """Grouped bar chart of median demean time by scenario and backend.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with columns: scenario, backend, demean_time_median.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    backends = sorted(summary_df["backend"].unique()) if "backend" in summary_df.columns else ["numba"]
    scenarios = summary_df["scenario"].unique()
    n_backends = len(backends)
    x = np.arange(len(scenarios))
    width = 0.8 / n_backends
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_backends))

    for i, backend in enumerate(backends):
        if "backend" in summary_df.columns:
            mask = summary_df["backend"] == backend
            times = summary_df[mask].set_index("scenario").reindex(scenarios)["demean_time_median"]
        else:
            times = summary_df.set_index("scenario").reindex(scenarios)["demean_time_median"]

        offset = (i - (n_backends - 1) / 2) * width
        bars = ax.bar(
            x + offset, times, width,
            label=backend, color=colors[i], edgecolor="black", alpha=0.85,
        )
        for bar, t in zip(bars, times):
            if pd.notna(t):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * summary_df["demean_time_median"].max(),
                    f"{t:.2f}s",
                    ha="center", va="bottom", fontsize=8, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Median Time (seconds)")
    ax.set_xlabel("Scenario")
    ax.set_title(title)
    ax.legend(title="Backend")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scenario_obs(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str = "Number of Observations by Scenario",
) -> None:
    """Bar chart of observation count by scenario.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with columns: scenario, n_obs.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Deduplicate across backends for obs count
    obs_df = summary_df.drop_duplicates(subset=["scenario"])[["scenario", "n_obs"]]
    scenarios = obs_df["scenario"]
    obs = obs_df["n_obs"]

    bars = ax.bar(scenarios, obs, color="coral", edgecolor="black", alpha=0.8)

    for bar, n in zip(bars, obs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * obs.max(),
            f"{n:,.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Number of Observations")
    ax.set_xlabel("Scenario")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_sweep(
    results_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    output_path: Path,
    title: str = "Parameter Sweep",
) -> None:
    """Line plot for a parameter sweep: time and connected set fraction vs parameter.

    If multiple backends are present, plots one line per backend.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with columns including x_col, demean_time_seconds,
        connected_set_fraction, and optionally backend.
    x_col : str
        Column name for the x-axis parameter.
    x_label : str
        Human-readable label for the x-axis.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    has_backends = "backend" in results_df.columns
    backends = sorted(results_df["backend"].unique()) if has_backends else ["numba"]
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(backends)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for i, backend in enumerate(backends):
        if has_backends:
            subset = results_df[results_df["backend"] == backend]
        else:
            subset = results_df

        agg = (
            subset.groupby(x_col)
            .agg(
                time_median=("demean_time_seconds", "median"),
                time_min=("demean_time_seconds", "min"),
                time_max=("demean_time_seconds", "max"),
                csf_median=("connected_set_fraction", "median"),
            )
            .reset_index()
            .sort_values(x_col)
        )

        ax1.plot(
            agg[x_col], agg["time_median"], "o-",
            color=colors[i], linewidth=2, label=backend,
        )
        ax1.fill_between(
            agg[x_col], agg["time_min"], agg["time_max"],
            alpha=0.15, color=colors[i],
        )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Demean Time (seconds)")
    ax1.set_title(f"{title}: Time")
    ax1.legend(title="Backend")
    ax1.grid(alpha=0.3)

    # Connected set fraction (same across backends, just plot once)
    csf_agg = (
        results_df.groupby(x_col)
        .agg(csf_median=("connected_set_fraction", "median"))
        .reset_index()
        .sort_values(x_col)
    )
    ax2.plot(csf_agg[x_col], csf_agg["csf_median"], "s-", color="darkgreen", linewidth=2)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Connected Set Fraction")
    ax2.set_title(f"{title}: Connected Set")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_time_vs_connected_set(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Demean Time vs Connected Set Fraction",
) -> None:
    """Scatter plot of demean time vs connected set fraction across all runs.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with columns: demean_time_seconds, connected_set_fraction,
        scenario, and optionally backend.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    has_backend = "backend" in results_df.columns
    if has_backend:
        groups = results_df.groupby(["scenario", "backend"])
        labels = [f"{s} ({b})" for s, b in groups.groups.keys()]
    else:
        groups = results_df.groupby("scenario")
        labels = list(groups.groups.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for (name, subset), color, label in zip(groups, colors, labels):
        ax.scatter(
            subset["connected_set_fraction"],
            subset["demean_time_seconds"],
            label=label,
            color=color,
            s=60, alpha=0.7,
            edgecolors="black", linewidth=0.5,
        )

    ax.set_xlabel("Connected Set Fraction")
    ax.set_ylabel("Demean Time (seconds)")
    ax.set_title(title)
    ax.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_time_vs_nobs(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Demean Time vs Number of Observations",
) -> None:
    """Scatter plot of demean time vs observation count.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    has_backend = "backend" in results_df.columns
    if has_backend:
        groups = results_df.groupby(["scenario", "backend"])
        labels = [f"{s} ({b})" for s, b in groups.groups.keys()]
    else:
        groups = results_df.groupby("scenario")
        labels = list(groups.groups.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for (name, subset), color, label in zip(groups, colors, labels):
        ax.scatter(
            subset["n_obs"],
            subset["demean_time_seconds"],
            label=label,
            color=color,
            s=60, alpha=0.7,
            edgecolors="black", linewidth=0.5,
        )

    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Demean Time (seconds)")
    ax.set_title(title)
    ax.legend(title="Scenario", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_iterations_vs_time(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Iterations vs Demean Time",
) -> None:
    """Scatter plot of iteration count vs wall-clock time by backend.

    Visually separates algorithmic improvements (fewer iterations)
    from implementation speedups (faster per-iteration time).

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with columns: demean_n_iterations, demean_time_seconds,
        backend.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    # Filter to rows with iteration data
    plot_df = results_df.dropna(subset=["demean_n_iterations"]).copy()
    if plot_df.empty:
        print(f"Skipped (no iteration data): {output_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    backends = sorted(plot_df["backend"].unique())
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(backends)))

    for i, backend in enumerate(backends):
        subset = plot_df[plot_df["backend"] == backend]
        ax.scatter(
            subset["demean_n_iterations"],
            subset["demean_time_seconds"],
            label=backend,
            color=colors[i],
            s=60, alpha=0.7,
            edgecolors="black", linewidth=0.5,
        )

    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Demean Time (seconds)")
    ax.set_title(title)
    ax.legend(title="Backend")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def format_baseline_comparison_table(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    tablefmt: str = "github",
) -> str:
    """Format a comparison table between current results and a saved baseline.

    Parameters
    ----------
    current_df : pd.DataFrame
        Summary from summarize_results() for the current run.
    baseline_df : pd.DataFrame
        Summary from a previously saved baseline.
    tablefmt : str
        Table format for tabulate.

    Returns
    -------
    str
        Formatted comparison table string.
    """
    merge_keys = ["scenario", "backend"]
    merged = current_df.merge(
        baseline_df,
        on=merge_keys,
        how="left",
        suffixes=("_current", "_baseline"),
    )

    rows = []
    for _, row in merged.iterrows():
        baseline_time = row.get("demean_time_median_baseline")
        current_time = row.get("demean_time_median_current")
        has_baseline = pd.notna(baseline_time)

        entry = {
            "Scenario": row["scenario"],
            "Backend": row["backend"],
            "Baseline (s)": f"{baseline_time:.3f}" if has_baseline else "-",
            "Current (s)": f"{current_time:.3f}" if pd.notna(current_time) else "-",
        }

        if has_baseline and pd.notna(current_time) and current_time > 0:
            speedup = baseline_time / current_time
            delta_pct = (baseline_time - current_time) / baseline_time * 100
            entry["Speedup"] = f"{speedup:.2f}x"
            entry["Delta %"] = f"{delta_pct:+.1f}%"
        else:
            entry["Speedup"] = "new" if not has_baseline else "-"
            entry["Delta %"] = "new" if not has_baseline else "-"

        # Add iteration delta if both sides have iteration data
        baseline_iters = row.get("demean_n_iterations_median_baseline")
        current_iters = row.get("demean_n_iterations_median_current")
        has_iters = pd.notna(baseline_iters) and pd.notna(current_iters)
        if has_iters:
            entry["Baseline Iters"] = f"{baseline_iters:.0f}"
            entry["Current Iters"] = f"{current_iters:.0f}"
            if baseline_iters > 0:
                iter_delta = (baseline_iters - current_iters) / baseline_iters * 100
                entry["Iter Delta %"] = f"{iter_delta:+.1f}%"
            else:
                entry["Iter Delta %"] = "-"

        rows.append(entry)

    if not rows:
        return "No matching entries to compare."

    result_df = pd.DataFrame(rows)

    # Compute geometric mean speedup for matched rows
    speedup_vals = []
    for _, row in merged.iterrows():
        bt = row.get("demean_time_median_baseline")
        ct = row.get("demean_time_median_current")
        if pd.notna(bt) and pd.notna(ct) and ct > 0:
            speedup_vals.append(bt / ct)

    parts = [tabulate(result_df, headers="keys", tablefmt=tablefmt, showindex=False)]

    if speedup_vals:
        geo_mean = np.exp(np.mean(np.log(speedup_vals)))
        parts.append(f"\nGeometric mean speedup: {geo_mean:.2f}x")

    return "\n".join(parts)


def plot_baseline_comparison(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: Path,
    baseline_name: str = "baseline",
) -> None:
    """Grouped bar chart comparing current results against a saved baseline.

    Only plots entries present in both current and baseline runs (inner merge).

    Parameters
    ----------
    current_df : pd.DataFrame
        Summary from summarize_results() for the current run.
    baseline_df : pd.DataFrame
        Summary from a previously saved baseline.
    output_path : Path
        Where to save the PNG file.
    baseline_name : str
        Name of the baseline for the legend.
    """
    merge_keys = ["scenario", "backend"]
    merged = current_df.merge(
        baseline_df,
        on=merge_keys,
        how="inner",
        suffixes=("_current", "_baseline"),
    )

    if merged.empty:
        print(f"Skipped (no overlapping entries): {output_path}")
        return

    labels = [
        f"{row['scenario']}\n({row['backend']})" for _, row in merged.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))

    baseline_times = merged["demean_time_median_baseline"]
    current_times = merged["demean_time_median_current"]

    bars_base = ax.bar(
        x - width / 2, baseline_times, width,
        label=f"Baseline ({baseline_name})", color="#6baed6",
        edgecolor="black", alpha=0.85,
    )
    bars_curr = ax.bar(
        x + width / 2, current_times, width,
        label="Current", color="#fd8d3c",
        edgecolor="black", alpha=0.85,
    )

    for bar, t in zip(bars_base, baseline_times):
        if pd.notna(t):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}s", ha="center", va="bottom", fontsize=8,
            )
    for bar, t in zip(bars_curr, current_times):
        if pd.notna(t):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}s", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Median Time (seconds)")
    ax.set_title(f"Baseline Comparison: {baseline_name} vs Current")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_features_sweep(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "Demean Time vs Number of Features",
) -> None:
    """Line plot of demean time vs n_features with one line per backend.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results with columns: n_features, demean_time_seconds, backend.
    output_path : Path
        Where to save the PNG file.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    has_backends = "backend" in results_df.columns
    backends = sorted(results_df["backend"].unique()) if has_backends else ["numba"]
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(backends)))

    for i, backend in enumerate(backends):
        if has_backends:
            subset = results_df[results_df["backend"] == backend]
        else:
            subset = results_df

        agg = (
            subset.groupby("n_features")
            .agg(
                time_median=("demean_time_seconds", "median"),
                time_min=("demean_time_seconds", "min"),
                time_max=("demean_time_seconds", "max"),
            )
            .reset_index()
            .sort_values("n_features")
        )

        ax.plot(
            agg["n_features"], agg["time_median"], "o-",
            color=colors[i], linewidth=2, label=backend,
        )
        ax.fill_between(
            agg["n_features"], agg["time_min"], agg["time_max"],
            alpha=0.15, color=colors[i],
        )

    ax.set_xlabel("Number of Features (columns)")
    ax.set_ylabel("Demean Time (seconds)")
    ax.set_title(title)
    ax.legend(title="Backend")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
