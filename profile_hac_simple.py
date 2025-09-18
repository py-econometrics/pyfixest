#!/usr/bin/env python3
"""
JIT-aware profiling for HAC optimization.

This script properly profiles numba-compiled functions by:
1. Warming up JIT compilation first
2. Using correct dtypes for numba
3. Timing only the compiled functions
4. Multiple runs for statistical accuracy

Usage:
    pixi shell --environment dev
    python profile_hac_simple.py
"""

import time
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(name):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{name:35s}: {elapsed:8.4f}s")


def warmup_jit_functions():
    """CRITICAL: Warm up JIT compilation before profiling."""
    print("Warming up JIT compilation...")

    from pyfixest.estimation.vcov_utils import _nw_meat_panel, _dk_meat_panel, _hac_meat_loop, _get_bartlett_weights

    # Small warmup data with correct dtypes
    np.random.seed(42)
    scores = np.ascontiguousarray(np.random.normal(0, 1, (100, 3)).astype(np.float64))
    panel_arr = np.ascontiguousarray(np.repeat(np.arange(10), 10).astype(np.int64))
    time_arr = np.ascontiguousarray(np.tile(np.arange(10), 10).astype(np.int64))

    # Sort arrays
    sort_idx = np.lexsort((time_arr, panel_arr))
    scores = scores[sort_idx]
    panel_arr = panel_arr[sort_idx]
    time_arr = time_arr[sort_idx]

    # Calculate metadata
    unique_panels, starts, counts = np.unique(panel_arr, return_index=True, return_counts=True)
    starts = starts.astype(np.int64)
    counts = counts.astype(np.int64)

    # Warm up NW (triggers compilation)
    try:
        _ = _nw_meat_panel(
            scores=scores,
            time_arr=time_arr,
            panel_arr=panel_arr,
            starts=starts,
            counts=counts,
            lag=2,
            balanced=True
        )
        print("  NW warmed up ✓")
    except Exception as e:
        print(f"  NW warmup failed: {e}")

    # Warm up DK
    try:
        unique_times, idx = np.unique(time_arr, return_index=True)
        idx = idx.astype(np.int64)
        _ = _dk_meat_panel(
            scores=scores,
            time_arr=time_arr,
            idx=idx,
            lag=2
        )
        print("  DK warmed up ✓")
    except Exception as e:
        print(f"  DK warmup failed: {e}")

    # Warm up HAC loop
    try:
        weights = _get_bartlett_weights(2)
        scores_small = scores[:10]  # Just 10 time periods
        _ = _hac_meat_loop(
            scores=scores_small,
            weights=weights,
            time_periods=10,
            k=3,
            lag=2
        )
        print("  HAC loop warmed up ✓")
    except Exception as e:
        print(f"  HAC loop warmup failed: {e}")

    print("JIT warmup complete!\n")


def setup_test_data(n_obs, k, n_panels, n_time):
    """Generate test data with correct dtypes for numba."""
    np.random.seed(42)

    # Generate scores with correct dtype
    scores = np.ascontiguousarray(
        np.random.normal(0, 1, (n_obs, k)).astype(np.float64)
    )

    # Panel and time arrays with correct dtypes
    panel_arr = np.ascontiguousarray(
        np.repeat(np.arange(n_panels), n_time).astype(np.int64)
    )
    time_arr = np.ascontiguousarray(
        np.tile(np.arange(n_time), n_panels).astype(np.int64)
    )

    # Sort by panel then time
    sort_idx = np.lexsort((time_arr, panel_arr))
    scores = scores[sort_idx]
    panel_arr = panel_arr[sort_idx]
    time_arr = time_arr[sort_idx]

    # Calculate panel metadata with correct dtypes
    unique_panels, starts, counts = np.unique(panel_arr, return_index=True, return_counts=True)
    starts = starts.astype(np.int64)
    counts = counts.astype(np.int64)

    # Time metadata for DK
    unique_times, time_idx = np.unique(time_arr, return_index=True)
    time_idx = time_idx.astype(np.int64)

    return {
        'scores': scores,
        'panel_arr': panel_arr,
        'time_arr': time_arr,
        'starts': starts,
        'counts': counts,
        'time_idx': time_idx,
        'n_panels': n_panels,
        'n_time': n_time
    }


def profile_newey_west_compiled():
    """Profile compiled Newey-West functions."""
    print("Newey-West JIT-Compiled Profiling")
    print("=" * 50)

    from pyfixest.estimation.vcov_utils import _nw_meat_panel

    configs = [
        {'n_obs': 20000, 'k': 5, 'n_panels': 2000, 'n_time': 10, 'lag': 3},
        {'n_obs': 50000, 'k': 8, 'n_panels': 5000, 'n_time': 10, 'lag': 5},
        {'n_obs': 100000, 'k': 12, 'n_panels': 10000, 'n_time': 10, 'lag': 7},
        {'n_obs': 200000, 'k': 15, 'n_panels': 20000, 'n_time': 10, 'lag': 10},
    ]

    for i, config in enumerate(configs):
        print(f"\n--- Config {i+1}: N={config['n_obs']}, K={config['k']}, Lag={config['lag']} ---")

        data = setup_test_data(
            config['n_obs'], config['k'],
            config['n_panels'], config['n_time']
        )

        # Multiple runs for statistical accuracy
        n_runs = 5

        # Test balanced panel (optimized path)
        times_balanced = []
        for run in range(n_runs):
            start = time.perf_counter()
            result_balanced = _nw_meat_panel(
                scores=data['scores'],
                time_arr=data['time_arr'],
                panel_arr=data['panel_arr'],
                starts=data['starts'],
                counts=data['counts'],
                lag=config['lag'],
                balanced=True
            )
            times_balanced.append(time.perf_counter() - start)

        avg_balanced = np.mean(times_balanced)
        std_balanced = np.std(times_balanced)
        print(f"NW Balanced:     {avg_balanced:.4f}s ± {std_balanced:.4f}s")

        # Test unbalanced panel (general path)
        times_unbalanced = []
        for run in range(n_runs):
            start = time.perf_counter()
            result_unbalanced = _nw_meat_panel(
                scores=data['scores'],
                time_arr=data['time_arr'],
                panel_arr=data['panel_arr'],
                starts=data['starts'],
                counts=data['counts'],
                lag=config['lag'],
                balanced=False
            )
            times_unbalanced.append(time.perf_counter() - start)

        avg_unbalanced = np.mean(times_unbalanced)
        std_unbalanced = np.std(times_unbalanced)
        print(f"NW Unbalanced:   {avg_unbalanced:.4f}s ± {std_unbalanced:.4f}s")

        # Performance ratio
        speedup = avg_unbalanced / avg_balanced
        print(f"Balanced speedup: {speedup:.2f}x")

        # Verify results are similar
        max_diff = np.max(np.abs(result_balanced - result_unbalanced))
        print(f"Max difference:   {max_diff:.2e}")


def profile_driscoll_kraay_compiled():
    """Profile compiled Driscoll-Kraay functions."""
    print("\n\nDriscoll-Kraay JIT-Compiled Profiling")
    print("=" * 50)

    from pyfixest.estimation.vcov_utils import _dk_meat_panel, _hac_meat_loop, _get_bartlett_weights

    configs = [
        {'n_obs': 2000, 'k': 5, 'n_panels': 200, 'n_time': 10, 'lag': 3},
        {'n_obs': 5000, 'k': 8, 'n_panels': 500, 'n_time': 10, 'lag': 5},
        {'n_obs': 10000, 'k': 12, 'n_panels': 1000, 'n_time': 10, 'lag': 7},
        {'n_obs': 20000, 'k': 15, 'n_panels': 2000, 'n_time': 10, 'lag': 10},
    ]

    for i, config in enumerate(configs):
        print(f"\n--- Config {i+1}: N={config['n_obs']}, K={config['k']}, Lag={config['lag']} ---")

        data = setup_test_data(
            config['n_obs'], config['k'],
            config['n_panels'], config['n_time']
        )

        n_runs = 5

        # Full DK computation
        times_dk_total = []
        for run in range(n_runs):
            start = time.perf_counter()
            dk_result = _dk_meat_panel(
                scores=data['scores'],
                time_arr=data['time_arr'],
                idx=data['time_idx'],
                lag=config['lag']
            )
            times_dk_total.append(time.perf_counter() - start)

        avg_dk_total = np.mean(times_dk_total)
        std_dk_total = np.std(times_dk_total)
        print(f"DK Total:        {avg_dk_total:.4f}s ± {std_dk_total:.4f}s")

        # Break down into components

        # 1. Time aggregation step
        times_aggregation = []
        for run in range(n_runs):
            start = time.perf_counter()
            scores_time = np.zeros((len(data['time_idx']), config['k']))
            for t in range(len(data['time_idx']) - 1):
                scores_time[t, :] = data['scores'][data['time_idx'][t]:data['time_idx'][t + 1], :].sum(axis=0)
            scores_time[-1, :] = data['scores'][data['time_idx'][-1]:, :].sum(axis=0)
            times_aggregation.append(time.perf_counter() - start)

        avg_aggregation = np.mean(times_aggregation)
        std_aggregation = np.std(times_aggregation)
        print(f"  Aggregation:   {avg_aggregation:.4f}s ± {std_aggregation:.4f}s ({100*avg_aggregation/avg_dk_total:.1f}%)")

        # 2. HAC computation step
        weights = _get_bartlett_weights(config['lag'])
        times_hac = []
        for run in range(n_runs):
            start = time.perf_counter()
            hac_result = _hac_meat_loop(
                scores=scores_time,
                weights=weights,
                time_periods=len(data['time_idx']),
                k=config['k'],
                lag=config['lag']
            )
            times_hac.append(time.perf_counter() - start)

        avg_hac = np.mean(times_hac)
        std_hac = np.std(times_hac)
        print(f"  HAC Loop:      {avg_hac:.4f}s ± {std_hac:.4f}s ({100*avg_hac/avg_dk_total:.1f}%)")

        # Verify breakdown sums correctly
        breakdown_total = avg_aggregation + avg_hac
        overhead = avg_dk_total - breakdown_total
        print(f"  Overhead:      {overhead:.4f}s ({100*overhead/avg_dk_total:.1f}%)")


def profile_scalability():
    """Test how performance scales with problem size."""
    print("\n\nScalability Analysis")
    print("=" * 50)

    from pyfixest.estimation.vcov_utils import _nw_meat_panel

    # Test different problem sizes
    base_config = {'n_panels': 100, 'n_time': 10, 'lag': 5}

    # Scale by number of observations
    print("\nScaling by observations:")
    for n_obs in [1000, 2000, 5000, 10000, 20000]:
        k = 8  # Fixed number of variables
        n_panels = n_obs // base_config['n_time']

        data = setup_test_data(n_obs, k, n_panels, base_config['n_time'])

        start = time.perf_counter()
        _nw_meat_panel(
            scores=data['scores'],
            time_arr=data['time_arr'],
            panel_arr=data['panel_arr'],
            starts=data['starts'],
            counts=data['counts'],
            lag=base_config['lag'],
            balanced=True
        )
        elapsed = time.perf_counter() - start

        print(f"  N={n_obs:5d}: {elapsed:.4f}s ({elapsed*1000/n_obs:.2f}ms per 1000 obs)")

    # Scale by number of variables
    print("\nScaling by variables:")
    n_obs = 5000
    n_panels = 500
    for k in [3, 5, 8, 12, 16, 20]:
        data = setup_test_data(n_obs, k, n_panels, base_config['n_time'])

        start = time.perf_counter()
        _nw_meat_panel(
            scores=data['scores'],
            time_arr=data['time_arr'],
            panel_arr=data['panel_arr'],
            starts=data['starts'],
            counts=data['counts'],
            lag=base_config['lag'],
            balanced=True
        )
        elapsed = time.perf_counter() - start

        print(f"  K={k:2d}: {elapsed:.4f}s ({elapsed*1000/(k*k):.2f}ms per K²)")


def main():
    """Run JIT-aware profiling."""
    print("JIT-Compiled HAC Function Profiling")
    print("=" * 60)

    # CRITICAL: Warm up JIT compilation first
    warmup_jit_functions()

    # Profile the compiled functions
    profile_newey_west_compiled()
    profile_driscoll_kraay_compiled()
    profile_scalability()

    print("\n" + "=" * 60)
    print("JIT profiling complete!")
    print("\nKey optimization insights:")
    print("1. Compare balanced vs unbalanced performance")
    print("2. Identify bottleneck: aggregation vs HAC computation")
    print("3. Check scaling behavior with N and K")
    print("4. Look for consistent timings (low std dev)")


if __name__ == "__main__":
    main()