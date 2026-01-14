"""
Reproduce multicollinearity detection bug with cupy/scipy backend.

The issue: When covariates are perfectly spanned by fixed effects,
the LSMR-based demeaning (cupy/scipy) leaves small residuals that
are not detected by the collinearity check for large datasets.
"""

import numpy as np
import pandas as pd
import pyfixest as pf

# Set seed for reproducibility
np.random.seed(42)

# Create a dataset where i(year) indicators are perfectly spanned by year^id fixed effects
N_ids = 1000
N_years = 10
N = N_ids * N_years

# Create panel data
ids = np.repeat(np.arange(N_ids), N_years)
years = np.tile(np.arange(N_years), N_ids)

# Create outcome with true effect from treatment and fixed effects
id_effects = np.random.normal(0, 10, N_ids)[ids]
year_effects = np.random.normal(0, 5, N_years)[years]
# year^id effects (interaction)
year_id_effects = np.random.normal(0, 2, (N_ids, N_years))
year_id_effect = year_id_effects[ids, years]

# Treatment variable (varies within id-year cells)
treatment = np.random.normal(0, 1, N)

# True coefficient
beta_treatment = 5.0

# Outcome
y = beta_treatment * treatment + id_effects + year_id_effect + np.random.normal(0, 1, N)

# Create DataFrame
df = pd.DataFrame({
    "y": y,
    "treatment": treatment,
    "id": ids,
    "year": years,
})

print("=" * 70)
print("MULTICOLLINEARITY DETECTION BUG REPRODUCTION")
print("=" * 70)
print(f"\nDataset: N={N:,} observations, {N_ids} ids, {N_years} years")
print(f"True treatment coefficient: {beta_treatment}")
print("\nModel: y ~ treatment + i(year, ref=0) | id + year^id")
print("Note: i(year) indicators are PERFECTLY SPANNED by year^id fixed effects!")
print("      These should be detected as collinear and dropped.\n")

# Formula with collinear terms
# i(year, ref=0) creates year indicators, but year^id FE already absorbs year variation
formula = "y ~ treatment + i(year, ref=0) | id + year^id"

# Test 1: Default cupy tolerance
print("-" * 70)
print("TEST 1: CuPy backend with DEFAULT LSMR tolerance (1e-8)")
print("-" * 70)

try:
    fit_default = pf.feols(
        formula,
        data=df,
        vcov="hetero",
        demeaner_backend="cupy",
    )
    print(f"Coefficients estimated: {len(fit_default.coef())}")
    print(f"Collinear vars dropped: {fit_default._collin_vars}")
    print("\nCoefficients:")
    for name, coef in fit_default.coef().items():
        se = fit_default.se()[name]
        print(f"  {name}: {coef:.6f} (se: {se:.6f})")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Tighter cupy tolerance
print("\n" + "-" * 70)
print("TEST 2: CuPy backend with TIGHT LSMR tolerance (1e-12)")
print("-" * 70)

# Monkey-patch the CupyFWLDemeaner to use tighter tolerance
from pyfixest.estimation.cupy.demean_cupy_ import CupyFWLDemeaner

# Store original __init__
_original_init = CupyFWLDemeaner.__init__

def _patched_init(self, use_gpu=None, solver_atol=1e-12, solver_btol=1e-12,
                  solver_maxiter=None, warn_on_cpu_fallback=True, dtype=np.float64):
    _original_init(self, use_gpu, solver_atol, solver_btol, solver_maxiter,
                   warn_on_cpu_fallback, dtype)

CupyFWLDemeaner.__init__ = _patched_init

try:
    fit_tight = pf.feols(
        formula,
        data=df,
        vcov="hetero",
        demeaner_backend="cupy",
    )
    print(f"Coefficients estimated: {len(fit_tight.coef())}")
    print(f"Collinear vars dropped: {fit_tight._collin_vars}")
    print("\nCoefficients:")
    for name, coef in fit_tight.coef().items():
        se = fit_tight.se()[name]
        print(f"  {name}: {coef:.6f} (se: {se:.6f})")
except Exception as e:
    print(f"Error: {e}")

# Restore original
CupyFWLDemeaner.__init__ = _original_init

# Test 3: Numba backend (should work correctly)
print("\n" + "-" * 70)
print("TEST 3: Numba backend (reference - should detect collinearity)")
print("-" * 70)

try:
    fit_numba = pf.feols(
        formula,
        data=df,
        vcov="hetero",
        demeaner_backend="numba",
    )
    print(f"Coefficients estimated: {len(fit_numba.coef())}")
    print(f"Collinear vars dropped: {fit_numba._collin_vars}")
    print("\nCoefficients:")
    for name, coef in fit_numba.coef().items():
        se = fit_numba.se()[name]
        print(f"  {name}: {coef:.6f} (se: {se:.6f})")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Rust backend (should work correctly)
print("\n" + "-" * 70)
print("TEST 4: Rust backend (reference - should detect collinearity)")
print("-" * 70)

try:
    fit_rust = pf.feols(
        formula,
        data=df,
        vcov="hetero",
        demeaner_backend="rust",
    )
    print(f"Coefficients estimated: {len(fit_rust.coef())}")
    print(f"Collinear vars dropped: {fit_rust._collin_vars}")
    print("\nCoefficients:")
    for name, coef in fit_rust.coef().items():
        se = fit_rust.se()[name]
        print(f"  {name}: {coef:.6f} (se: {se:.6f})")
except Exception as e:
    print(f"Error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Expected behavior:
- All i(year) indicators should be dropped as collinear with year^id FE
- Only 'treatment' coefficient should be estimated (~5.0)

If cupy with default tolerance shows multiple coefficients but numba/rust
show only 'treatment', then the bug is confirmed: LSMR tolerance is too
loose for the collinearity check to detect perfectly spanned covariates.
""")
