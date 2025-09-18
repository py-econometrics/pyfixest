#!/usr/bin/env python3
"""
Profiling script for HAC variance-covariance matrix computation.

Usage:
    pixi shell --environment dev
    python profile_hac.py
"""

import time
import numpy as np
import pandas as pd
import pyfixest as pf
from contextlib import contextmanager
import cProfile
import pstats
import io


@contextmanager
def timer(name):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.4f}s")


def generate_test_data(n_obs=10000, n_panels=1000, n_time=50, n_vars=5, balanced=True):
    """Generate test data for profiling."""
    np.random.seed(42)

    if balanced:
        # Balanced panel
        panel_ids = np.repeat(np.arange(n_panels), n_time)
        time_ids = np.tile(np.arange(n_time), n_panels)
        n_obs = n_panels * n_time
    else:
        # Unbalanced panel
        panel_ids = []
        time_ids = []
        for panel in range(n_panels):
            # Random number of observations per panel
            n_obs_panel = np.random.randint(max(1, n_time//2), n_time + 1)
            panel_ids.extend([panel] * n_obs_panel)
            time_ids.extend(np.random.choice(n_time, n_obs_panel, replace=False))

        panel_ids = np.array(panel_ids)
        time_ids = np.array(time_ids)
        n_obs = len(panel_ids)

    # Generate X variables
    X_data = {}
    for i in range(n_vars):
        X_data[f'x{i+1}'] = np.random.normal(0, 1, n_obs)

    # Generate Y with some correlation structure
    y = np.sum([X_data[f'x{i+1}'] for i in range(n_vars)], axis=0) + np.random.normal(0, 1, n_obs)

    # Create DataFrame
    data = pd.DataFrame({
        'y': y,
        'panel_id': panel_ids,
        'time_id': time_ids,
        **X_data
    })

    return data


def profile_pyfixest_hac(data, vcov_type='newey_west'):
    """Profile pyfixest HAC computation."""
    print(f"\n=== Profiling {vcov_type.upper()} ===")
    print(f"Data shape: {data.shape}")
    print(f"Panels: {data['panel_id'].nunique()}")
    print(f"Time periods: {data['time_id'].nunique()}")

    # Formula
    formula = 'y ~ ' + ' + '.join([f'x{i+1}' for i in range(5)])

    with timer("Total pyfixest time"):
        if vcov_type == 'newey_west':
            fit = pf.feols(
                formula,
                data=data,
                vcov={'NW': {'maxlag': 5}},
                cluster={'panel': 'panel_id', 'time': 'time_id'}
            )
        elif vcov_type == 'driscoll_kraay':
            fit = pf.feols(
                formula,
                data=data,
                vcov={'DK': {'maxlag': 5}},
                cluster={'time': 'time_id'}
            )

    return fit


def detailed_hac_profiling():
    """Run detailed profiling of HAC functions."""
    print("\n=== Detailed HAC Function Profiling ===")

    # Import the functions we want to profile
    from pyfixest.estimation.vcov_utils import _nw_meat_panel, _dk_meat_panel

    # Generate test data directly for the functions
    np.random.seed(42)
    n_obs, k = 5000, 8
    scores = np.random.normal(0, 1, (n_obs, k))

    # Panel data setup
    n_panels = 500
    n_time = 10
    panel_arr = np.repeat(np.arange(n_panels), n_time)
    time_arr = np.tile(np.arange(n_time), n_panels)

    # Sort by panel then time
    sort_idx = np.lexsort((time_arr, panel_arr))
    scores = scores[sort_idx]
    panel_arr = panel_arr[sort_idx]
    time_arr = time_arr[sort_idx]

    # Calculate starts and counts for panels
    unique_panels, starts, counts = np.unique(panel_arr, return_index=True, return_counts=True)

    print(f"Scores shape: {scores.shape}")
    print(f"N panels: {len(unique_panels)}")
    print(f"N time periods: {len(np.unique(time_arr))}")

    # Profile Newey-West
    print("\n--- Newey-West Profiling ---")
    with timer("NW meat computation"):
        nw_result = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=5,
            balanced=None
        )

    # Profile Driscoll-Kraay
    print("\n--- Driscoll-Kraay Profiling ---")
    unique_times, idx = np.unique(time_arr, return_index=True)

    with timer("DK meat computation"):
        dk_result = _dk_meat_panel(
            scores=scores,
            time_arr=time_arr,
            idx=idx,
            lag=5
        )

    print(f"NW result shape: {nw_result.shape}")
    print(f"DK result shape: {dk_result.shape}")


def cprofile_analysis():
    """Run cProfile analysis on HAC computation."""
    print("\n=== cProfile Analysis ===")

    # Generate test data
    data = generate_test_data(n_obs=5000, n_panels=500, n_time=10, balanced=True)

    # Profile with cProfile
    pr = cProfile.Profile()
    pr.enable()

    # Run the computation
    fit = pf.feols(
        'y ~ x1 + x2 + x3 + x4 + x5',
        data=data,
        vcov={'NW': {'maxlag': 5}},
        cluster={'panel': 'panel_id', 'time': 'time_id'}
    )

    pr.disable()

    # Create stats object
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    print("Top 20 functions by cumulative time:")
    print(s.getvalue())

    # Focus on HAC-related functions
    print("\nHAC-related functions:")
    stats.print_stats('.*hac.*|.*meat.*|.*vcov.*', 15)


def compare_balanced_vs_unbalanced():
    """Compare performance of balanced vs unbalanced panels."""
    print("\n=== Balanced vs Unbalanced Comparison ===")

    # Test parameters
    test_configs = [
        {'n_panels': 200, 'n_time': 20, 'n_vars': 5},
        {'n_panels': 500, 'n_time': 10, 'n_vars': 8},
        {'n_panels': 1000, 'n_time': 5, 'n_vars': 3},
    ]

    for config in test_configs:
        print(f"\nConfig: {config}")

        # Balanced panel
        data_balanced = generate_test_data(
            n_panels=config['n_panels'],
            n_time=config['n_time'],
            n_vars=config['n_vars'],
            balanced=True
        )

        with timer(f"Balanced (N={len(data_balanced)})"):
            fit_balanced = profile_pyfixest_hac(data_balanced, 'newey_west')

        # Unbalanced panel
        data_unbalanced = generate_test_data(
            n_panels=config['n_panels'],
            n_time=config['n_time'],
            n_vars=config['n_vars'],
            balanced=False
        )

        with timer(f"Unbalanced (N={len(data_unbalanced)})"):
            fit_unbalanced = profile_pyfixest_hac(data_unbalanced, 'newey_west')


def memory_profiling():
    """Basic memory usage profiling."""
    print("\n=== Memory Usage Analysis ===")

    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        def get_memory_mb():
            return process.memory_info().rss / 1024 / 1024

        initial_memory = get_memory_mb()
        print(f"Initial memory: {initial_memory:.1f} MB")

        # Generate large dataset
        data = generate_test_data(n_obs=20000, n_panels=2000, n_time=10, balanced=True)
        after_data_memory = get_memory_mb()
        print(f"After data generation: {after_data_memory:.1f} MB (+{after_data_memory - initial_memory:.1f} MB)")

        # Run computation
        fit = profile_pyfixest_hac(data, 'newey_west')
        after_computation_memory = get_memory_mb()
        print(f"After computation: {after_computation_memory:.1f} MB (+{after_computation_memory - after_data_memory:.1f} MB)")

    except ImportError:
        print("psutil not available for memory profiling")


def main():
    """Run comprehensive profiling."""
    print("HAC Variance-Covariance Matrix Profiling")
    print("=" * 50)

    # Basic performance comparison
    compare_balanced_vs_unbalanced()

    # Detailed function profiling
    detailed_hac_profiling()

    # cProfile analysis
    cprofile_analysis()

    # Memory profiling
    memory_profiling()

    print("\n" + "=" * 50)
    print("Profiling complete!")
    print("\nOptimization suggestions:")
    print("1. Focus on the functions with highest cumulative time")
    print("2. Check if balanced vs unbalanced shows significant differences")
    print("3. Look for memory allocation hotspots")
    print("4. Consider the trade-offs between different algorithms")


if __name__ == "__main__":
    main()
