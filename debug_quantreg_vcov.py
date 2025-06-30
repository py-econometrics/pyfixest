import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
from scipy.stats import norm

# Generate simple test data
np.random.seed(4242)
N = 1000
data = pd.DataFrame({
    'Y': 1 + 2 * np.random.normal(size=N) + np.random.normal(size=N),
    'X1': np.random.normal(size=N)
})

quantile = 0.5
fml = "Y ~ X1"

# Fit models
fit_py = pf.quantreg(fml, data=data, vcov="iid", quantile=quantile,
                     ssc=pf.ssc(adj=False, cluster_adj=False))
fit_sm = smf.quantreg(fml, data=data).fit(q=quantile, vcov="iid", kernel="cos", bandwidth="hsheather")

print("=== Comparing pyfixest vs statsmodels ===")
print(f"pyfixest SE: {fit_py.se().to_numpy()}")
print(f"statsmodels SE: {fit_sm.bse.to_numpy()}")
print(f"Ratio (pf/sm): {fit_py.se().to_numpy() / fit_sm.bse.to_numpy()}")

# Let's debug the variance computation step by step
print("\n=== Debugging pyfixest variance computation ===")

# Access pyfixest internal components
X = fit_py._X
Y = fit_py._Y
u_hat = fit_py._u_hat
q = fit_py._quantile
N = fit_py._N

# Hall-Sheather bandwidth
from pyfixest.estimation.quantreg.utils import get_hall_sheather_bandwidth
h = get_hall_sheather_bandwidth(q=q, N=N)
print(f"Hall-Sheather bandwidth h: {h}")

# Kernel bandwidth computation
rq = np.quantile(np.abs(u_hat), 0.75) - np.quantile(np.abs(u_hat), 0.25)
sigma = np.std(Y)
hk = np.minimum(sigma, rq / 1.34) * (norm.ppf(q + h) - norm.ppf(q - h))
print(f"Kernel bandwidth hk: {hk}")
print(f"sigma: {sigma}, rq: {rq}")
print(f"norm.ppf factor: {norm.ppf(q + h) - norm.ppf(q - h)}")

# Density estimation
f_hat_0 = np.sum(np.abs(u_hat) < hk) / (2 * N * hk)
sparsity = 1 / f_hat_0
print(f"Density estimate f_hat_0: {f_hat_0}")
print(f"Sparsity (1/f_hat_0): {sparsity}")
print(f"Number of residuals < hk: {np.sum(np.abs(u_hat) < hk)}")

# Final variance computation
XXinv = np.linalg.inv(X.T @ X)
vcov_manual = q * (1-q) * sparsity ** 2 * XXinv
se_manual = np.sqrt(np.diag(vcov_manual))
print(f"Manual SE computation: {se_manual}")
print(f"pyfixest SE: {fit_py.se().to_numpy()}")

print("\n=== Trying alternative density estimations ===")

# Alternative 1: Different scaling in denominator
f_hat_0_alt1 = np.sum(np.abs(u_hat) < hk) / (N * hk)  # Remove factor of 2
sparsity_alt1 = 1 / f_hat_0_alt1
vcov_alt1 = q * (1-q) * sparsity_alt1 ** 2 * XXinv
se_alt1 = np.sqrt(np.diag(vcov_alt1))
print(f"Alternative 1 (no factor 2): {se_alt1}")
print(f"Ratio to statsmodels: {se_alt1 / fit_sm.bse.to_numpy()}")

# Alternative 2: Different sparsity power
vcov_alt2 = q * (1-q) * sparsity * XXinv  # sparsity^1 instead of sparsity^2
se_alt2 = np.sqrt(np.diag(vcov_alt2))
print(f"Alternative 2 (sparsity^1): {se_alt2}")
print(f"Ratio to statsmodels: {se_alt2 / fit_sm.bse.to_numpy()}")