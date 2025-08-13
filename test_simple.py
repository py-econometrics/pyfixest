import numpy as np
import pandas as pd
import pyfixest as pf

# Create simple test data
rng = np.random.default_rng(12345)
nobs = 200
df = pd.DataFrame()
df["treatment"] = rng.normal(size=nobs)
df["med1"] = df["treatment"] * 0.5 + rng.normal(loc=0, scale=0.2, size=nobs)
df["med2"] = df["treatment"] * 0.3 + rng.normal(loc=0, scale=0.2, size=nobs)
df["med3"] = -df["treatment"] * 0.4 + rng.normal(loc=0, scale=0.2, size=nobs)
df["y"] = (df["treatment"] * 1.0 + df["med1"] * 0.8 + 
           df["med2"] * (-0.6) + df["med3"] * 0.9 + 
           rng.normal(loc=0, scale=0.3, size=nobs))

# Fit and decompose
fit = pf.feols("y ~ treatment + med1 + med2 + med3", data=df)
gb = fit.decompose(decomp_var="treatment", only_coef=True)

print("Testing updated coefplot...")
gb.coefplot(figsize=(10, 6), title="Test: Sorted Negative/Positive Effects")
print("✓ Coefplot completed successfully!")

# Show the results
results = gb.tidy()
levels = results[results["panels"] == "Levels (units)"]
print("\nDecomposition results:")
print(levels)
