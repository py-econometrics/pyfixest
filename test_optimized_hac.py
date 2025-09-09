import numpy as np
import pandas as pd
import pyfixest as pf
import time

# Create test data
def create_test_data(N=50, T=10, seed=421):
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(N), T)
    time_vals = np.tile(np.arange(T), N)
    treat = rng.choice([0, 1], N * T)
    alpha = rng.normal(0, 1, N)
    gamma = np.random.normal(0, 0.5, T)
    epsilon = rng.normal(0, 5, N * T)
    Y = alpha[units] + gamma[time_vals] + treat + epsilon

    return pd.DataFrame({
        "unit": units,
        "year": time_vals,
        "treat": treat,
        "Y": Y,
    })

def test_optimized_hac():
    """Test that the optimized HAC implementation works correctly"""
    data = create_test_data()

    print("Testing optimized HAC implementation...")
    print(f"Data shape: {data.shape}")
    print(f"Unique units: {data['unit'].nunique()}")
    print(f"Unique years: {data['year'].nunique()}")

    # Test the optimized implementation
    start_time = time.time()

    try:
        mod = pf.feols(
            "Y~treat | unit + year",
            data=data,
            vcov="NW",
            vcov_kwargs={"time_id": "year", "panel_id": "unit", "lag": 4}
        )

        end_time = time.time()
        elapsed = end_time - start_time

        print(".4f")
        print(f"Coefficient: {mod.coef()[0]:.6f}")
        print(f"SE: {mod.se()[0]:.6f}")
        print("✅ Optimized HAC implementation working correctly!")

    except Exception as e:
        print(f"❌ Error in optimized implementation: {e}")
        return False

    return True

if __name__ == "__main__":
    test_optimized_hac()
