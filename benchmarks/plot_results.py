"""
Generate plots from saved benchmark CSV files.

Reads CSVs from results/individual_benchmarks/ and writes PNGs to
results/plots/.

Usage:
    python -m benchmarks.plot_results [--baseline NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmarks.bench import summarize_results
from benchmarks.plot import (
    plot_baseline_comparison,
    plot_features_sweep,
    plot_iterations_vs_time,
    plot_scenario_obs,
    plot_scenario_times,
    plot_sweep,
    plot_time_vs_connected_set,
    plot_time_vs_nobs,
)

RESULTS_DIR = Path(__file__).parent / "results"
CSV_DIR = RESULTS_DIR / "individual_benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"
BASELINES_DIR = RESULTS_DIR / "baselines"

SWEEP_META: dict[str, dict[str, str]] = {
    "mobility": {"x_col": "p_move", "x_label": "Move Probability (p_move)"},
    "pareto": {"x_col": "pareto_shape", "x_label": "Pareto Shape (theta)"},
    "cluster": {"x_col": "scenario", "x_label": "Cluster Config"},
    "group_count": {"x_col": "n_workers", "x_label": "Number of Workers"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from benchmark CSVs.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, metavar="NAME",
        help="Also plot a baseline comparison against NAME.",
    )
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # --- Scenario plots ---
    scenario_csv = CSV_DIR / "scenario_results.csv"
    if scenario_csv.exists():
        df = pd.read_csv(scenario_csv)
        summary = summarize_results(df)
        plot_scenario_times(summary, PLOTS_DIR / "scenario_times.png")
        plot_scenario_obs(summary, PLOTS_DIR / "scenario_obs.png")
        generated += ["scenario_times.png", "scenario_obs.png"]

    # --- Sweep plots ---
    for sweep_name, meta in SWEEP_META.items():
        sweep_csv = CSV_DIR / f"sweep_{sweep_name}.csv"
        if not sweep_csv.exists():
            continue
        df = pd.read_csv(sweep_csv)
        plot_sweep(
            df,
            x_col=meta["x_col"],
            x_label=meta["x_label"],
            output_path=PLOTS_DIR / f"sweep_{sweep_name}.png",
            title=f"Sweep: {sweep_name}",
        )
        generated.append(f"sweep_{sweep_name}.png")

    # --- Features sweep plot ---
    features_csv = CSV_DIR / "sweep_features.csv"
    if features_csv.exists():
        df = pd.read_csv(features_csv)
        plot_features_sweep(df, output_path=PLOTS_DIR / "sweep_features.png")
        generated.append("sweep_features.png")

    # --- Combined plots (from all_results.csv) ---
    all_csv = CSV_DIR / "all_results.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
        plot_time_vs_connected_set(df, PLOTS_DIR / "time_vs_connected_set.png")
        plot_time_vs_nobs(df, PLOTS_DIR / "time_vs_nobs.png")
        generated += ["time_vs_connected_set.png", "time_vs_nobs.png"]

        if (
            "demean_n_iterations" in df.columns
            and df["demean_n_iterations"].notna().any()
        ):
            plot_iterations_vs_time(df, PLOTS_DIR / "iterations_vs_time.png")
            generated.append("iterations_vs_time.png")

    # --- Baseline comparison plot ---
    if args.baseline and all_csv.exists():
        baseline_path = BASELINES_DIR / f"{args.baseline}.csv"
        if baseline_path.exists():
            df = pd.read_csv(all_csv)
            current_summary = summarize_results(df)
            baseline_df = pd.read_csv(baseline_path)
            plot_baseline_comparison(
                current_summary, baseline_df,
                output_path=PLOTS_DIR / f"baseline_comparison_{args.baseline}.png",
                baseline_name=args.baseline,
            )
            generated.append(f"baseline_comparison_{args.baseline}.png")
        else:
            print(f"Warning: baseline {args.baseline!r} not found at {baseline_path}")

    if generated:
        print(f"Generated {len(generated)} plots in {PLOTS_DIR}/")
    else:
        print(f"No CSVs found in {CSV_DIR}/ -- run benchmarks first.")


if __name__ == "__main__":
    main()
