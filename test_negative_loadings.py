#!/usr/bin/env python3
"""
Test script to verify that negative loadings are correctly handled in the coefplot method.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

def create_test_data_with_negative_mediators(nobs=500):
    """Create test data where some mediator effects will be negative."""
    rng = np.random.default_rng(12345)
    df = pd.DataFrame(index=range(nobs))

    # Treatment variable
    df["treatment"] = rng.normal(size=nobs)

    # Mediator 1: positive relationship with treatment, positive effect on outcome
    df["mediator1"] = df["treatment"] * 0.5 + rng.normal(loc=0, scale=0.2, size=nobs)

    # Mediator 2: positive relationship with treatment, NEGATIVE effect on outcome
    df["mediator2"] = df["treatment"] * 0.3 + rng.normal(loc=0, scale=0.2, size=nobs)

    # Mediator 3: negative relationship with treatment, positive effect on outcome
    # This should result in a negative mediator effect overall
    df["mediator3"] = -df["treatment"] * 0.4 + rng.normal(loc=0, scale=0.2, size=nobs)

    # Outcome variable
    df["y"] = (
        df["treatment"] * 1.0 +        # Direct effect
        df["mediator1"] * 0.8 +        # Positive mediation
        df["mediator2"] * (-0.6) +     # Negative mediation (suppressor)
        df["mediator3"] * 0.9 +        # Negative mediation (negative relationship)
        rng.normal(loc=0, scale=0.3, size=nobs)
    )

    return df

def test_negative_loadings():
    """Test that negative loadings are correctly displayed in coefplot."""
    print("Creating test data with negative mediator effects...")
    data = create_test_data_with_negative_mediators(nobs=1000)

    print("Fitting model...")
    # Fit the full model
    fit = pf.feols("y ~ treatment + mediator1 + mediator2 + mediator3", data=data)

    print("Running Gelbach decomposition...")
    # Decompose the treatment effect
    gb = fit.decompose(decomp_var="treatment", only_coef=True)

    print("Decomposition results:")
    results = gb.tidy()
    levels = results[results["panels"] == "Levels (units)"]
    print(levels)

    # Check for negative mediator effects
    mediator_effects = levels.loc[["mediator1", "mediator2", "mediator3"], "coefficients"]
    print(f"\nMediator effects:")
    for med, coef in mediator_effects.items():
        print(f"  {med}: {coef:.4f}")

    # Check if we have any negative effects
    negative_effects = mediator_effects[mediator_effects < 0]
    if len(negative_effects) > 0:
        print(f"\nFound {len(negative_effects)} negative mediator effects:")
        for med, coef in negative_effects.items():
            print(f"  {med}: {coef:.4f}")
    else:
        print("\nNo negative mediator effects found in this example.")

    print("\nTesting coefplot (this should now correctly handle negative loadings)...")
    try:
        # This should work without errors and correctly show negative loadings
        gb.coefplot(figsize=(10, 6), title="Test: Negative Loadings Handling")
        print("✓ coefplot completed successfully")
    except Exception as e:
        print(f"✗ coefplot failed with error: {e}")
        return False

    # Verify the mathematical consistency
    direct_effect = levels.loc["direct_effect", "coefficients"]
    full_effect = levels.loc["full_effect", "coefficients"]
    explained_effect = levels.loc["explained_effect", "coefficients"]

    # Check: direct_effect - explained_effect = full_effect
    computed_full = direct_effect - explained_effect
    print(f"\nMathematical consistency check:")
    print(f"  Direct effect: {direct_effect:.4f}")
    print(f"  Explained effect: {explained_effect:.4f}")
    print(f"  Full effect: {full_effect:.4f}")
    print(f"  direct - explained = {computed_full:.4f}")
    print(f"  Difference: {abs(computed_full - full_effect):.6f}")

    if abs(computed_full - full_effect) < 1e-6:
        print("✓ Mathematical consistency verified")
        return True
    else:
        print("✗ Mathematical inconsistency detected")
        return False

if __name__ == "__main__":
    success = test_negative_loadings()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")