"""
Main entry point for running demeaning benchmarks.

Defines benchmark scenarios (easy through extreme) and single-parameter
sweeps, then orchestrates data generation, benchmarking, and plotting.

Usage:
    python -m benchmarks.run_benchmarks [--scenarios] [--sweeps] [--all]
        [--feols] [--reps N] [--n-features N] [--fe-columns COL ...]
        [--sweep-name NAME]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from benchmarks.bench import (
    BenchmarkResult,
    results_to_dataframe,
    run_benchmark,
    summarize_results,
)
from benchmarks.dgp import DGPConfig
from benchmarks.plot import (
    format_baseline_comparison_table,
    format_comparison_table,
    format_summary_table,
    plot_baseline_comparison,
    plot_features_sweep,
    plot_iterations_vs_time,
    plot_scenario_obs,
    plot_scenario_times,
    plot_sweep,
    plot_time_vs_connected_set,
    plot_time_vs_nobs,
    save_table,
)

RESULTS_DIR = Path(__file__).parent / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, DGPConfig] = {
    "easy": DGPConfig(
        n_workers=10_000,
        n_firms=1_000,
        n_years=10,
        p_move=0.15,
        pareto_shape=5.0,
        n_clusters=1,
        p_between_cluster=1.0,
        p_observe=1.0,
        selection_worker=0.0,
        p_survive=1.0,
        selection_firm=0.0,
        seed=42,
    ),
    "medium": DGPConfig(
        n_workers=50_000,
        n_firms=5_000,
        n_years=15,
        p_move=0.05,
        pareto_shape=2.0,
        n_clusters=5,
        p_between_cluster=0.3,
        p_observe=0.8,
        selection_worker=0.5,
        p_survive=0.95,
        selection_firm=1.0,
        spell_concentration=3.0,
        seed=42,
    ),
    "hard": DGPConfig(
        n_workers=100_000,
        n_firms=10_000,
        n_years=20,
        p_move=0.02,
        pareto_shape=1.0,
        n_clusters=10,
        p_between_cluster=0.1,
        p_observe=0.6,
        selection_worker=1.0,
        p_survive=0.90,
        selection_firm=2.0,
        spell_concentration=5.0,
        sorting_wf=0.3,
        seed=42,
    ),
    "large": DGPConfig(
        n_workers=150_000,
        n_firms=15_000,
        n_years=15,
        p_move=0.02,
        pareto_shape=2.0,
        n_clusters=10,
        p_between_cluster=0.1,
        p_observe=0.8,
        selection_worker=0.5,
        p_survive=0.95,
        selection_firm=1.0,
        spell_concentration=3.0,
        seed=42,
    ),
}

# ---------------------------------------------------------------------------
# Parameter sweep definitions
# ---------------------------------------------------------------------------

_SWEEP_DEFAULTS = asdict(SCENARIOS["large"])


def _sweep_config(**overrides: object) -> DGPConfig:
    """Create a DGPConfig from large defaults with overrides."""
    params = {**_SWEEP_DEFAULTS, **overrides}
    return DGPConfig(**params)


MOBILITY_SWEEP = {
    f"mobility_p{p}": _sweep_config(p_move=p)
    for p in [0.01, 0.10, 0.30]
}

PARETO_SWEEP = {
    f"pareto_th{th}": _sweep_config(pareto_shape=th)
    for th in [1.0, 3.0, 10.0]
}

CLUSTER_SWEEP = {
    f"cluster_k{k}_p{p}": _sweep_config(n_clusters=k, p_between_cluster=p)
    for k, p in [(1, 1.0), (5, 0.1), (20, 0.05)]
}

GROUP_COUNT_SWEEP = {
    f"groups_w{nw}_f{nf}": _sweep_config(
        n_workers=nw, n_firms=nf, n_years=10, p_observe=1.0, p_survive=1.0,
    )
    for nw, nf in [
        (3_000, 300),
        (30_000, 3_000),
        (150_000, 15_000),
    ]
}

ALL_SWEEPS: dict[str, dict[str, DGPConfig]] = {
    "mobility": MOBILITY_SWEEP,
    "pareto": PARETO_SWEEP,
    "cluster": CLUSTER_SWEEP,
    "group_count": GROUP_COUNT_SWEEP,
}

SWEEP_META: dict[str, dict[str, str]] = {
    "mobility": {"x_col": "p_move", "x_label": "Move Probability (p_move)"},
    "pareto": {"x_col": "pareto_shape", "x_label": "Pareto Shape (theta)"},
    "cluster": {"x_col": "scenario", "x_label": "Cluster Config"},
    "group_count": {"x_col": "n_workers", "x_label": "Number of Workers"},
}

# Features sweep: same DGP config, varying n_features
FEATURES_SWEEP_VALUES = [1, 5, 20]


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def _save_baseline(summary_df: pd.DataFrame, name: str) -> Path:
    """Save a summary DataFrame as a named baseline CSV.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary from summarize_results().
    name : str
        Baseline name (used as filename without extension).

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{name}.csv"
    summary_df.to_csv(path, index=False)
    print(f"Saved baseline: {path}")
    return path


def _load_baseline(name: str) -> pd.DataFrame:
    """Load a previously saved baseline CSV.

    Parameters
    ----------
    name : str
        Baseline name (filename without extension).

    Returns
    -------
    pd.DataFrame
        The saved summary DataFrame.

    Raises
    ------
    FileNotFoundError
        If the baseline file does not exist.
    """
    path = BASELINES_DIR / f"{name}.csv"
    if not path.exists():
        available = sorted(p.stem for p in BASELINES_DIR.glob("*.csv")) if BASELINES_DIR.exists() else []
        available_str = ", ".join(available) if available else "(none)"
        raise FileNotFoundError(
            f"Baseline {name!r} not found at {path}. "
            f"Available baselines: {available_str}"
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_scenario_suite(
    scenarios: dict[str, DGPConfig],
    n_reps: int,
    backends: list[str],
    run_feols: bool,
    n_features: int = 1,
    fe_columns: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run a set of named scenarios and return all results."""
    all_results: list[BenchmarkResult] = []
    total = len(scenarios)
    for i, (name, config) in enumerate(scenarios.items(), 1):
        print(f"\n{'='*60}")
        print(f"Scenario {i}/{total}: {name}")
        print(f"{'='*60}")
        results = run_benchmark(
            config, name,
            n_repetitions=n_reps, backends=backends, run_feols=run_feols,
            n_features=n_features, fe_columns=fe_columns,
        )
        all_results.extend(results)
    return all_results


def _add_config_cols(df: pd.DataFrame, results: list[BenchmarkResult]) -> pd.DataFrame:
    """Add DGP config columns to results DataFrame for sweep plotting."""
    config_rows = []
    for r in results:
        config_rows.append({
            "scenario": r.scenario_name,
            "p_move": r.config.p_move,
            "pareto_shape": r.config.pareto_shape,
            "n_clusters": r.config.n_clusters,
            "p_between_cluster": r.config.p_between_cluster,
            "p_observe": r.config.p_observe,
            "p_survive": r.config.p_survive,
            "selection_firm": r.config.selection_firm,
            "sorting_wf": r.config.sorting_wf,
            "n_workers": r.config.n_workers,
            "n_firms": r.config.n_firms,
        })
    config_df = pd.DataFrame(config_rows)
    config_df.index = df.index
    for col in config_df.columns:
        if col not in df.columns:
            df[col] = config_df[col]
        elif col in ("n_workers", "n_firms"):
            # Overwrite with config values for group_count sweep
            df[col] = config_df[col]
    return df


def _run_features_sweep(
    n_reps: int,
    backends: list[str],
    fe_columns: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run the features sweep: same medium DGP, varying n_features."""
    all_results: list[BenchmarkResult] = []
    config = SCENARIOS["large"]

    for n_feat in FEATURES_SWEEP_VALUES:
        name = f"features_{n_feat}"
        print(f"\n{'='*60}")
        print(f"Features sweep: n_features={n_feat}")
        print(f"{'='*60}")
        results = run_benchmark(
            config, name,
            n_repetitions=n_reps, backends=backends, run_feols=False,
            n_features=n_feat, fe_columns=fe_columns,
        )
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run benchmarks and produce output."""
    parser = argparse.ArgumentParser(description="PyFixest Demeaning Benchmarks")
    parser.add_argument(
        "--scenarios", action="store_true",
        help="Run the main scenario suite (easy/medium/hard/extreme).",
    )
    parser.add_argument(
        "--sweeps", action="store_true",
        help="Run all parameter sweeps.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run everything (scenarios + sweeps).",
    )
    parser.add_argument(
        "--backends", nargs="+", default=["numba"],
        help="Demeaner backends to benchmark. Default: numba. "
             "Options: numba, rust, scipy, jax.",
    )
    parser.add_argument(
        "--feols", action="store_true",
        help="Also benchmark feols() (slower).",
    )
    parser.add_argument(
        "--reps", type=int, default=3,
        help="Number of repetitions per scenario. Default 3.",
    )
    parser.add_argument(
        "--n-features", type=int, default=1,
        help="Number of columns to demean. Default 1.",
    )
    parser.add_argument(
        "--fe-columns", nargs="+", default=None,
        help="Which FE columns to use. Default: worker_id firm_id year.",
    )
    parser.add_argument(
        "--sweep-name", type=str, default=None,
        help="Run only a single named sweep (e.g. 'mobility', 'features', 'group_count').",
    )
    parser.add_argument(
        "--save-baseline", type=str, default=None, metavar="NAME",
        help="Save combined summary as a named baseline to benchmarks/results/baselines/NAME.csv.",
    )
    parser.add_argument(
        "--baseline", type=str, default=None, metavar="NAME",
        help="Compare current results against a previously saved baseline.",
    )
    args = parser.parse_args()

    if not (args.scenarios or args.sweeps or args.all):
        args.all = True

    backends = args.backends
    print(f"Backends: {backends}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []

    # --- Main scenarios ---
    if args.scenarios or args.all:
        print("\n" + "=" * 60)
        print("MAIN SCENARIOS")
        print("=" * 60)
        scenario_results = _run_scenario_suite(
            SCENARIOS, args.reps, backends, args.feols,
            n_features=args.n_features, fe_columns=args.fe_columns,
        )
        all_results.extend(scenario_results)

        df = results_to_dataframe(scenario_results)
        df.to_csv(RESULTS_DIR / "scenario_results.csv", index=False)

        summary = summarize_results(df)
        summary.to_csv(RESULTS_DIR / "scenario_summary.csv", index=False)

        print("\n--- Scenario Summary ---")
        print(format_summary_table(summary))
        if len(backends) > 1:
            print("\n--- Backend Comparison ---")
            print(format_comparison_table(summary))
        save_table(summary, RESULTS_DIR / "scenario_table.txt")

        plot_scenario_times(summary, RESULTS_DIR / "scenario_times.png")
        plot_scenario_obs(summary, RESULTS_DIR / "scenario_obs.png")

    # --- Parameter sweeps ---
    if args.sweeps or args.all:
        # Determine which sweeps to run
        if args.sweep_name == "features":
            sweeps_to_run = {}
        elif args.sweep_name is not None:
            if args.sweep_name not in ALL_SWEEPS:
                available = list(ALL_SWEEPS.keys()) + ["features"]
                parser.error(
                    f"Unknown sweep {args.sweep_name!r}. "
                    f"Available: {available}"
                )
            sweeps_to_run = {args.sweep_name: ALL_SWEEPS[args.sweep_name]}
        else:
            sweeps_to_run = ALL_SWEEPS

        for sweep_name, sweep_configs in sweeps_to_run.items():
            print(f"\n{'='*60}")
            print(f"SWEEP: {sweep_name}")
            print(f"{'='*60}")

            sweep_results = _run_scenario_suite(
                sweep_configs, args.reps, backends, args.feols,
                n_features=args.n_features, fe_columns=args.fe_columns,
            )
            all_results.extend(sweep_results)

            df = results_to_dataframe(sweep_results)
            df = _add_config_cols(df, sweep_results)
            df.to_csv(RESULTS_DIR / f"sweep_{sweep_name}.csv", index=False)

            sweep_summary = summarize_results(df)
            print(f"\n--- {sweep_name} sweep summary ---")
            print(format_summary_table(sweep_summary))

            meta = SWEEP_META[sweep_name]
            plot_sweep(
                df,
                x_col=meta["x_col"],
                x_label=meta["x_label"],
                output_path=RESULTS_DIR / f"sweep_{sweep_name}.png",
                title=f"Sweep: {sweep_name}",
            )

        # --- Features sweep ---
        run_features = (
            args.sweep_name is None or args.sweep_name == "features"
        )
        if run_features:
            print(f"\n{'='*60}")
            print("SWEEP: features")
            print(f"{'='*60}")

            features_results = _run_features_sweep(
                args.reps, backends, fe_columns=args.fe_columns,
            )
            all_results.extend(features_results)

            df = results_to_dataframe(features_results)
            df.to_csv(RESULTS_DIR / "sweep_features.csv", index=False)

            features_summary = summarize_results(df)
            print("\n--- features sweep summary ---")
            print(format_summary_table(features_summary))

            plot_features_sweep(
                df, output_path=RESULTS_DIR / "sweep_features.png",
            )

    # --- Combined plots ---
    if all_results:
        combined_df = results_to_dataframe(all_results)
        combined_df = _add_config_cols(combined_df, all_results)
        combined_df.to_csv(RESULTS_DIR / "all_results.csv", index=False)

        plot_time_vs_connected_set(
            combined_df, RESULTS_DIR / "time_vs_connected_set.png"
        )
        plot_time_vs_nobs(
            combined_df, RESULTS_DIR / "time_vs_nobs.png"
        )

        # Plot iterations vs time if any backend returned iteration counts
        if (
            "demean_n_iterations" in combined_df.columns
            and combined_df["demean_n_iterations"].notna().any()
        ):
            plot_iterations_vs_time(
                combined_df, RESULTS_DIR / "iterations_vs_time.png"
            )

    # --- Baseline save/compare ---
    if all_results:
        combined_summary = summarize_results(combined_df)

        if args.save_baseline:
            _save_baseline(combined_summary, args.save_baseline)

        if args.baseline:
            try:
                baseline_df = _load_baseline(args.baseline)
                print(f"\n--- Baseline Comparison (vs {args.baseline!r}) ---")
                comparison_text = format_baseline_comparison_table(
                    combined_summary, baseline_df,
                )
                print(comparison_text)

                # Save comparison text
                comp_txt_path = RESULTS_DIR / f"baseline_comparison_{args.baseline}.txt"
                comp_txt_path.write_text(comparison_text)
                print(f"Saved: {comp_txt_path}")

                # Save comparison plot
                comp_png_path = RESULTS_DIR / f"baseline_comparison_{args.baseline}.png"
                plot_baseline_comparison(
                    combined_summary, baseline_df,
                    output_path=comp_png_path,
                    baseline_name=args.baseline,
                )
            except FileNotFoundError as e:
                print(f"\nWarning: {e}")
                print("Skipping baseline comparison.")
    elif args.save_baseline:
        print("\nWarning: No results to save as baseline.")

    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
