#!/usr/bin/env python3
"""
Quick test script to verify HAC functions work and get baseline timings.

Usage:
    pixi shell --environment dev
    python quick_hac_test.py
"""

import time
import numpy as np
import pandas as pd
import pyfixest as pf


def quick_test():
    """Quick test of HAC functionality."""
    print("Quick HAC Test")
    print("=" * 30)

    # Generate simple test data
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'y': np.random.normal(0, 1, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'panel_id': np.repeat(np.arange(100), 10),
        'time_id': np.tile(np.arange(10), 100)
    })

    print(f"Data shape: {data.shape}")
    print(f"Panels: {data['panel_id'].nunique()}")
    print(f"Time periods: {data['time_id'].nunique()}")

    # Test Newey-West
    start_time = time.time()
    fit_nw = pf.feols(
        'y ~ x1 + x2',
        data=data,
        vcov="NW",
        vcov_kwargs={'panel_id': 'panel_id', 'time_id': 'time_id', "lag": 3}
    )
    nw_time = time.time() - start_time

    print(f"\nNewey-West time: {nw_time:.4f}s")
    print("Coefficients:", fit_nw.coef().values)
    print("Standard errors:", fit_nw.se().values)

    # Test Driscoll-Kraay
    start_time = time.time()
    fit_dk = pf.feols(
        'y ~ x1 + x2',
        data=data,
        vcov="DK",
        vcov_kwargs={'time_id': 'time_id', "panel_id": "panel_id", "lag": 3}
    )
    dk_time = time.time() - start_time

    print(f"\nDriscoll-Kraay time: {dk_time:.4f}s")
    print("Coefficients:", fit_dk.coef().values)
    print("Standard errors:", fit_dk.se().values)

    print(f"\nTest completed successfully!")
    print(f"Total time: {nw_time + dk_time:.4f}s")


if __name__ == "__main__":
    quick_test()
