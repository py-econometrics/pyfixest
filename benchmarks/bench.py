"""
Benchmark runner for PyFixest demeaning.

Times PyFixest's demean() function across multiple backends and optionally
feols() on synthetic three-way fixed effects panel data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from benchmarks.dgp import DGPConfig, ThreeWayFEData

# Valid backend names (subset of pyfixest's DemeanerBackendOptions that work on CPU)
CPU_BACKENDS = ("numba", "rust", "scipy", "jax")

# Default fixed effect columns used in benchmarks
DEFAULT_FE_COLUMNS = ["worker_id", "firm_id", "year"]


def _get_demean_func(backend: str) -> Callable:
    """Import and return the demean function for a given backend.

    Parameters
    ----------
    backend : str
        One of "numba", "rust", "scipy", "jax".

    Returns
    -------
    Callable
        The backend's demean function.
    """
    from pyfixest.estimation.backends import BACKENDS

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Available: {list(BACKENDS.keys())}"
        )
    return BACKENDS[backend]["demean"]


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes
    ----------
    config : DGPConfig
        The DGP configuration used.
    scenario_name : str
        Human-readable name for this scenario.
    backend : str
        Which demeaner backend was used.
    n_obs : int
        Number of observations in the generated data.
    n_workers : int
        Number of observed workers.
    n_firms : int
        Number of observed firms.
    n_years : int
        Number of years.
    connected_set_fraction : float
        Fraction of observations in the largest connected component.
    demean_time_seconds : float
        Wall-clock time for the demean() call.
    demean_converged : bool
        Whether demeaning converged.
    feols_time_seconds : float | None
        Wall-clock time for feols() call, if run.
    feols_converged : bool | None
        Whether feols converged, if run.
    demean_n_iterations : int | None
        Iteration count if the backend returns it.
    demean_time_per_iter : float | None
        Computed time / iterations when available.
    n_features : int
        Number of columns demeaned.
    n_factors : int
        Number of FE factors used.
    """

    config: DGPConfig
    scenario_name: str
    backend: str
    n_obs: int
    n_workers: int
    n_firms: int
    n_years: int
    connected_set_fraction: float
    demean_time_seconds: float
    demean_converged: bool
    feols_time_seconds: float | None = None
    feols_converged: bool | None = None
    demean_n_iterations: int | None = None
    demean_time_per_iter: float | None = None
    n_features: int = 1
    n_factors: int = 3


def _prepare_demean_inputs(
    df: pd.DataFrame,
    n_features: int = 1,
    fe_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare arrays for demean() from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column "y" and the specified FE columns.
    n_features : int
        Number of columns to demean. When >1, stacks y with
        deterministically-seeded random columns.
    fe_columns : list[str] or None
        Which columns to use as fixed effects. Defaults to
        ["worker_id", "firm_id", "year"].

    Returns
    -------
    y : np.ndarray of shape (n_obs, n_features)
    flist : np.ndarray of shape (n_obs, n_factors)
    weights : np.ndarray of shape (n_obs,)
    """
    if fe_columns is None:
        fe_columns = DEFAULT_FE_COLUMNS

    y_col = df["y"].values.astype(np.float64)

    if n_features == 1:
        y = y_col.reshape(-1, 1)
    else:
        rng = np.random.default_rng(seed=12345)
        extra = rng.standard_normal((len(df), n_features - 1))
        y = np.column_stack([y_col, extra])

    flist = df[fe_columns].values.astype(np.int64)
    weights = np.ones(len(df), dtype=np.float64)
    return y, flist, weights


def _warmup_backend(demean_func: Callable) -> None:
    """Run a tiny warmup call to trigger JIT compilation."""
    _y = np.ones((10, 1), dtype=np.float64)
    _fe = np.zeros((10, 2), dtype=np.int64)
    _fe[:5, 0] = 1
    _fe[5:, 1] = 1
    _w = np.ones(10, dtype=np.float64)
    try:
        demean_func(_y, _fe, _w, tol=1e-6, maxiter=10)
    except Exception:
        pass


def _run_demean(
    df: pd.DataFrame,
    backend: str = "numba",
    n_features: int = 1,
    fe_columns: list[str] | None = None,
) -> tuple[float, bool, int | None]:
    """Run a demean backend on the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have column "y" and the FE columns.
    backend : str
        Backend name (numba, rust, scipy, jax).
    n_features : int
        Number of columns to demean.
    fe_columns : list[str] or None
        Which columns to use as fixed effects.

    Returns
    -------
    elapsed : float
        Wall-clock seconds.
    converged : bool
        Whether the algorithm converged.
    n_iters : int or None
        Number of iterations if backend returns it, else None.
    """
    demean_func = _get_demean_func(backend)
    y, flist, weights = _prepare_demean_inputs(
        df, n_features=n_features, fe_columns=fe_columns,
    )

    _warmup_backend(demean_func)

    start = time.perf_counter()
    result = demean_func(y, flist, weights, tol=1e-08, maxiter=100_000)
    elapsed = time.perf_counter() - start

    # Handle both 2-tuple (result, converged) and
    # 3-tuple (result, converged, n_iterations) returns
    if len(result) == 3:
        _, converged, n_iters = result
        n_iters = int(n_iters)
    else:
        _, converged = result
        n_iters = None

    return elapsed, bool(converged), n_iters


def _run_feols(
    df: pd.DataFrame, backend: str = "numba"
) -> tuple[float, bool]:
    """Run PyFixest's feols() on the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: worker_id, firm_id, year, y.
    backend : str
        Demeaner backend to use with feols().

    Returns
    -------
    elapsed : float
        Wall-clock seconds.
    converged : bool
        Whether the estimation succeeded without error.
    """
    import pyfixest as pf

    df_copy = df.copy()
    for col in ["worker_id", "firm_id", "year"]:
        df_copy[col] = df_copy[col].astype("category")

    start = time.perf_counter()
    try:
        pf.feols(
            "y ~ 1 | worker_id + firm_id + year",
            data=df_copy,
            demeaner_backend=backend,
        )
        converged = True
    except ValueError:
        converged = False
    elapsed = time.perf_counter() - start

    return elapsed, converged


def run_benchmark(
    config: DGPConfig,
    scenario_name: str,
    n_repetitions: int = 3,
    backends: list[str] | None = None,
    run_feols: bool = False,
    n_features: int = 1,
    fe_columns: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run a benchmark scenario across one or more backends.

    Parameters
    ----------
    config : DGPConfig
        Configuration for the data generating process.
    scenario_name : str
        Human-readable name for this scenario.
    n_repetitions : int
        Number of repetitions to run. Default 3.
    backends : list[str] or None
        Backends to benchmark. Default ["numba"].
    run_feols : bool
        Whether to also benchmark feols(). Default False.
    n_features : int
        Number of columns to demean. Default 1.
    fe_columns : list[str] or None
        Which FE columns to use. Default ["worker_id", "firm_id", "year"].

    Returns
    -------
    list[BenchmarkResult]
        One result per (repetition, backend) combination.
    """
    if backends is None:
        backends = ["numba"]
    if fe_columns is None:
        fe_columns = DEFAULT_FE_COLUMNS

    n_factors = len(fe_columns)

    results: list[BenchmarkResult] = []

    # Generate data once with a fixed seed, then run n_repetitions timed calls
    print(
        f"  [{scenario_name}] Generating data (seed={config.seed})..."
    )

    dgp = ThreeWayFEData(config)
    dgp_result = dgp.simulate()

    if dgp_result.n_obs == 0:
        print(
            f"  [{scenario_name}] No observations generated, skipping."
        )
        return results

    print(
        f"  [{scenario_name}] {dgp_result.n_obs:,} obs, "
        f"{dgp_result.n_workers_observed:,} workers, "
        f"{dgp_result.n_firms_observed:,} firms."
    )

    for backend in backends:
        for rep in range(n_repetitions):
            print(
                f"  [{scenario_name}] Rep {rep + 1}/{n_repetitions}: "
                f"demean({backend}, {n_features} col(s))...",
                end="",
                flush=True,
            )
            demean_time, demean_conv, n_iters = _run_demean(
                dgp_result.data, backend,
                n_features=n_features, fe_columns=fe_columns,
            )
            time_per_iter = (
                demean_time / n_iters if n_iters is not None and n_iters > 0 else None
            )
            iters_str = f", iters={n_iters}" if n_iters is not None else ""
            print(f" {demean_time:.3f}s, converged={demean_conv}{iters_str}")

            feols_time = None
            feols_conv = None
            if run_feols:
                print(
                    f"  [{scenario_name}] Rep {rep + 1}: "
                    f"feols({backend})...",
                    end="",
                    flush=True,
                )
                feols_time, feols_conv = _run_feols(dgp_result.data, backend)
                print(f" {feols_time:.3f}s, converged={feols_conv}")

            results.append(
                BenchmarkResult(
                    config=config,
                    scenario_name=scenario_name,
                    backend=backend,
                    n_obs=dgp_result.n_obs,
                    n_workers=dgp_result.n_workers_observed,
                    n_firms=dgp_result.n_firms_observed,
                    n_years=config.n_years,
                    connected_set_fraction=dgp_result.connected_set_fraction,
                    demean_time_seconds=demean_time,
                    demean_converged=demean_conv,
                    feols_time_seconds=feols_time,
                    feols_converged=feols_conv,
                    demean_n_iterations=n_iters,
                    demean_time_per_iter=time_per_iter,
                    n_features=n_features,
                    n_factors=n_factors,
                )
            )

    return results


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    """Convert a list of BenchmarkResults to a DataFrame.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Results to convert.

    Returns
    -------
    pd.DataFrame
        One row per result with all benchmark metrics.
    """
    rows = []
    for r in results:
        rows.append({
            "scenario": r.scenario_name,
            "backend": r.backend,
            "n_obs": r.n_obs,
            "n_workers": r.n_workers,
            "n_firms": r.n_firms,
            "n_years": r.n_years,
            "connected_set_fraction": r.connected_set_fraction,
            "demean_time_seconds": r.demean_time_seconds,
            "demean_converged": r.demean_converged,
            "feols_time_seconds": r.feols_time_seconds,
            "feols_converged": r.feols_converged,
            "demean_n_iterations": r.demean_n_iterations,
            "demean_time_per_iter": r.demean_time_per_iter,
            "n_features": r.n_features,
            "n_factors": r.n_factors,
        })
    return pd.DataFrame(rows)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize benchmark results by scenario and backend.

    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results from results_to_dataframe().

    Returns
    -------
    pd.DataFrame
        One row per (scenario, backend) with median timing.
    """
    group_cols = ["scenario", "backend"]
    agg_dict = {
        "n_obs": ("n_obs", "median"),
        "n_workers": ("n_workers", "median"),
        "n_firms": ("n_firms", "median"),
        "connected_set_fraction": ("connected_set_fraction", "median"),
        "demean_time_median": ("demean_time_seconds", "median"),
        "demean_time_min": ("demean_time_seconds", "min"),
        "demean_time_max": ("demean_time_seconds", "max"),
        "demean_converged": ("demean_converged", "all"),
        "n_runs": ("demean_time_seconds", "count"),
    }

    # Add iteration aggregation when data is available
    if (
        "demean_n_iterations" in results_df.columns
        and results_df["demean_n_iterations"].notna().any()
    ):
        agg_dict["demean_n_iterations_median"] = ("demean_n_iterations", "median")

    if (
        "demean_time_per_iter" in results_df.columns
        and results_df["demean_time_per_iter"].notna().any()
    ):
        agg_dict["demean_time_per_iter_median"] = ("demean_time_per_iter", "median")

    # Add n_features and n_factors (take first since constant within group)
    if "n_features" in results_df.columns:
        agg_dict["n_features"] = ("n_features", "first")
    if "n_factors" in results_df.columns:
        agg_dict["n_factors"] = ("n_factors", "first")

    summary = (
        results_df.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
    )

    if (
        "feols_time_seconds" in results_df.columns
        and results_df["feols_time_seconds"].notna().any()
    ):
        feols_summary = (
            results_df.dropna(subset=["feols_time_seconds"])
            .groupby(group_cols)
            .agg(
                feols_time_median=("feols_time_seconds", "median"),
                feols_converged=("feols_converged", "all"),
            )
            .reset_index()
        )
        summary = summary.merge(feols_summary, on=group_cols, how="left")

    return summary
