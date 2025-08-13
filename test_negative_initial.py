import numpy as np
import pandas as pd
import pyfixest as pf

# Create test data with negative initial difference
rng = np.random.default_rng(54321)
nobs = 200
df = pd.DataFrame()

# Treatment variable
df["treatment"] = rng.normal(size=nobs)

# Create mediators that will result in negative direct effect
df["med1"] = df["treatment"] * 0.3 + rng.normal(loc=0, scale=0.2, size=nobs)
df["med2"] = df["treatment"] * 0.2 + rng.normal(loc=0, scale=0.2, size=nobs)
df["med3"] = -df["treatment"] * 0.4 + rng.normal(loc=0, scale=0.2, size=nobs)

# Outcome variable - designed to have negative direct effect
df["y"] = (
    df["treatment"] * (-0.8) +     # Negative direct effect
    df["med1"] * 0.6 +             # Positive mediation
    df["med2"] * (-0.3) +          # Negative mediation
    df["med3"] * 0.5 +             # This should be negative overall
    rng.normal(loc=0, scale=0.3, size=nobs)
)

# Fit and decompose
print("Fitting model with expected negative initial difference...")
fit = pf.feols("y ~ treatment + med1 + med2 + med3", data=df)
gb = fit.decompose(decomp_var="treatment", only_coef=True)

# Show the results
results = gb.tidy()
levels = results[results["panels"] == "Levels (units)"]
print("\nDecomposition results:")
print(levels)

direct_effect = levels.loc["direct_effect", "coefficients"]
print(f"\nDirect effect: {direct_effect:.4f}")

if direct_effect < 0:
    print("✓ Confirmed: Initial difference is negative")
    print("Testing mirrored waterfall chart...")
    gb.coefplot(figsize=(10, 6), title="Test: Negative Initial Difference (Mirrored)")
    print("✓ Negative initial difference chart completed!")
else:
    print("Initial difference is positive. Creating alternative test...")
    # If the random data didn't produce negative effect, force it
    print("Testing with positive initial difference first...")
    gb.coefplot(figsize=(10, 6), title="Test: Positive Initial Difference")
