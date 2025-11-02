import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import nbinom
from tqdm import tqdm

import pyfixest as pf

np.random.seed(42)

import cupy as cp

print(cp.ones(10).device)

# ============================================================================
# VISUALIZATION
# ============================================================================
import os

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80 + "\n")


def create_benchmark_plot(df, title_suffix, filename, facet_by_fixef=True):
    """Create a benchmark plot for either standard or complex data."""
    df = df.copy()
    df["G"] = df["G"].map({1: "n_fixef = 1", 2: "n_fixef = 2", 3: "n_fixef = 3"})
    df["n_obs"] = df["n_obs"].astype(str)

    # Dynamically determine unique values for order and hue_order
    n_obs_order = sorted(df["n_obs"].unique(), key=lambda x: int(x))
    demeaner_backend_order = df["demeaner_backend"].unique()

    custom_palette = sns.color_palette("coolwarm", n_colors=len(demeaner_backend_order))

    if facet_by_fixef:
        # Create the FacetGrid with reordered columns and rows
        g = sns.FacetGrid(
            df,
            col="G",  # G (n_fixef) increases left to right
            margin_titles=True,
            height=4,
            aspect=1.2,
            col_order=["n_fixef = 1", "n_fixef = 2", "n_fixef = 3"],
            sharey=False,
        )

        # Plot the bar chart for each facet with the custom palette
        g.map(
            sns.barplot,
            "n_obs",
            "full_feols_timing",
            "demeaner_backend",
            order=n_obs_order,
            hue_order=demeaner_backend_order,
            errorbar=None,
            palette=custom_palette,
        )

        # Add legend and adjust layout
        g.add_legend(title="Demeaner Backend")
        g.set_axis_labels("Number of Observations", "Runtime (seconds)")
        g.set_titles(col_template="{col_name}")
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Runtime vs Number of Observations by n_fixef ({title_suffix})")
    else:
        # Create a single plot without faceting
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=df,
            x="n_obs",
            y="full_feols_timing",
            hue="demeaner_backend",
            order=n_obs_order,
            hue_order=demeaner_backend_order,
            errorbar=None,
            palette=custom_palette,
            ax=ax,
        )

        ax.set_xlabel("Number of Observations", fontsize=12)
        ax.set_ylabel("Runtime (seconds)", fontsize=12)
        ax.set_title(
            f"Runtime vs Number of Observations ({title_suffix})", fontsize=14, pad=20
        )
        ax.legend(title="Demeaner Backend", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Create a mock FacetGrid-like object to maintain compatibility
        class SimplePlot:
            def __init__(self, fig):
                self.fig = fig

            def savefig(self, *args, **kwargs):
                self.fig.savefig(*args, **kwargs)

        g = SimplePlot(fig)

    # Save figure
    g.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved {title_suffix} plot to: {os.path.abspath(filename)}")

    return g


def generate_test_data(size: int, k: int = 2):
    """
    Generate benchmark data for pyfixest on GPU (similar to the R fixest benchmark data).

    Args:
        size (int): The number of observations in the data frame.
        k (int): The number of covariates in the data frame.

    Returns
    -------
        pd.DataFrame: The generated data frame for the given size.
    """
    # Constants
    all_n = [1000 * 10**i for i in range(5)]
    a = 1
    b = 0.05

    n = all_n[size - 1]

    dum_all = []
    nb_dum = [n // 20, int(np.sqrt(n)), int(n**0.33)]

    dum_all = np.zeros((n, 3))
    dum_all[:, 0] = np.random.choice(nb_dum[0], n, replace=True)
    dum_all[:, 1] = np.random.choice(nb_dum[1], n, replace=True)
    dum_all[:, 2] = np.random.choice(nb_dum[2], n, replace=True)
    dum_all = dum_all.astype(int)

    X1 = np.random.normal(size=n)
    X2 = X1**2

    mu = a * X1 + b * X2

    for m in range(3):
        coef_dum = np.random.normal(size=nb_dum[m])
        mu += coef_dum[dum_all[:, m]]

    mu = np.exp(mu)
    y = nbinom.rvs(0.5, 1 - (mu / (mu + 0.5)), size=n)

    X_full = np.column_stack((X1, X2))
    base = pd.DataFrame(
        {
            "y": y,
            "ln_y": np.log(y + 1),
            "X1": X1,
            "X2": X2,
        }
    )

    if k > 2:
        X = np.random.normal(size=(n, k - 2))
        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(3, k + 1, 1)])
        base = pd.concat([base, X_df], axis=1)
        X_full = np.column_stack((X_full, X))

    for m in range(3):
        base[f"dum_{m + 1}"] = dum_all[:, m]

    weights = np.random.uniform(0, 1, n)
    return base, y, X_full, dum_all, weights


def generate_complex_fixed_effects_data(n: int = 10**5):
    """
    Complex fixed effects example ported from fixest R-implementation:
    https://github.com/lrberge/fixest/blob/ac1be27fda5fc381c0128b861eaf5bda88af846c/_BENCHMARK/Data%20generation.R#L125 .

    Always generates:
    - 3 fixed effects: id_indiv, id_firm, id_year
    - 2 covariates: X1, X2
    - Variable n observations

    Args:
        n (int): Number of observations. Default is 10^5.

    Returns
    -------
        tuple: (pd.DataFrame, y, X_full, flist, weights)
    """
    rng = np.random.default_rng(42)
    nb_indiv = n // 20
    nb_firm = max(1, round(n / 160))
    nb_year = max(1, round(n**0.3))

    # Generate fixed effect IDs
    id_indiv = rng.choice(nb_indiv, n, replace=True)
    id_firm_base = rng.integers(0, 21, n) + np.maximum(1, id_indiv // 8 - 10)
    id_firm = np.minimum(id_firm_base, nb_firm - 1)
    id_year = rng.choice(nb_year, n, replace=True)

    # Create variables
    x1 = (
        5 * np.cos(id_indiv)
        + 5 * np.sin(id_firm)
        + 5 * np.sin(id_year)
        + rng.uniform(0, 1, n)
    )
    x2 = np.cos(id_indiv) + np.sin(id_firm) + np.sin(id_year) + rng.normal(0, 1, n)
    y = (
        3 * x1
        + 5 * x2
        + np.cos(id_indiv)
        + np.cos(id_firm) ** 2
        + np.sin(id_year)
        + rng.normal(0, 1, n)
    )

    # Build dataframe with exactly 2 covariates and 3 fixed effects
    base = pd.DataFrame(
        {
            "y": y,
            "X1": x1,
            "X2": x2,
            "id_indiv": id_indiv,
            "id_firm": id_firm,
            "id_year": id_year,
        }
    )

    X_full = np.column_stack([x1, x2, y])
    flist = np.column_stack([id_indiv, id_firm, id_year]).astype(np.uint64)
    weights = rng.uniform(0.5, 2.0, n)

    return base, y, X_full, flist, weights


df, Y, X, f, weights = generate_test_data(1)
m0 = pf.feols("ln_y ~ X1 | dum_1", df, demeaner_backend="rust")
m1 = pf.feols("ln_y ~ X1 | dum_1", df, demeaner_backend="cupy")


def run_standard_benchmark(
    fixed_effect,
    demeaner_backend,
    size=1,
    k=1,
    solver="np.linalg.solve",
    skip_demean_benchmark=True,
    nrep=3,
):
    """
    Run the fixest standard benchmark fixed effect models. This is the function the benchmarks
    will loop over.

    Args:
        fixed_effect (str): The fixed effect to use. Must be a list of variables as "dum_1", "dum_1+dum_2", or "dum_1+dum_2+dum_3", etc.
        demeaner_backend (str): The backend to use for demeaning. Must be "numba" or "jax".
        size (int): The size of the data to generate. Must be between 1 and 5. For 1, N = 1000, for 2, N = 10000, etc.
        k_vals (int): The number of covariates to generate.
        solver (str): The solver to use for the estimation. Must be "np.linalg.
        skip_demean_benchmark (bool): Whether to skip the "pure" demean benchmark. Default is True. Only the full call
            to feols is benchmarked.

    """
    assert fixed_effect in ["dum_1", "dum_1+dum_2", "dum_1+dum_2+dum_3"]

    # one fixed effect
    res = []

    fml_base = "ln_y ~ X1"
    fml = f"{fml_base} | {fixed_effect}"

    # warmup
    df, _, X, f, weights = generate_test_data(1)
    pf.feols(
        fml,
        data=df,
        demeaner_backend=demeaner_backend,
        store_data=False,
        copy_data=False,
        solver=solver,
        fixef_tol=1e-06,
        fixef_maxiter=500_000,
    )

    if k > 1:
        xfml = "+".join([f"X{i}" for i in range(2, k + 1, 1)])
        fml = f"{fml_base} + {xfml} | {fixed_effect}"
    else:
        fml = f"{fml_base} + X1 | {fixed_effect}"

    for rep in range(nrep):
        df, Y, X, f, weights = generate_test_data(size=size, k=k)

        tic1 = time.time()
        pf.feols(
            fml,
            data=df,
            demeaner_backend=demeaner_backend,
            store_data=False,
            copy_data=False,
            solver=solver,
            fixef_tol=1e-06,
            fixef_maxiter=500_000,
        )
        tic2 = time.time()

        full_feols_timing = tic2 - tic1

        demean_timing = np.nan
        if not skip_demean_benchmark:
            YX = np.column_stack((Y.reshape(-1, 1), X))
            tic3 = time.time()
            if demeaner_backend == "jax":
                _, _ = demean_jax(YX, f, weights, tol=1e-10)
            else:
                _, _ = demean(YX, f, weights, tol=1e-10)
            tic4 = time.time()
            demean_timing = tic4 - tic3

        res.append(
            pd.Series(
                {
                    "method": "feols",
                    "solver": solver,
                    "demeaner_backend": demeaner_backend,
                    "n_obs": df.shape[0],
                    "k": k,
                    "G": len(fixed_effect.split("+")),
                    "rep": rep + 1,
                    "full_feols_timing": full_feols_timing,
                    "demean_timing": demean_timing,
                }
            )
        )

    return pd.concat(res, axis=1).T


def run_complex_benchmark(
    demeaner_backend,
    n=10**5,
    solver="np.linalg.solve",
    nrep=3,
):
    """
    Run benchmarks on complex fixed effect models using the R fixest benchmark data structure.

    This benchmark always uses:
    - 3 fixed effects: id_indiv + id_firm + id_year
    - 2 covariates: X1 + X2

    Args:
        demeaner_backend (str): The backend to use for demeaning.
        n (int): The number of observations to generate.
        solver (str): The solver to use for the estimation.
        nrep (int): Number of repetitions.

    Returns
    -------
        pd.DataFrame: Benchmark results.
    """
    res = []

    # Fixed formula: always 2 covariates and 3 fixed effects
    fml = "y ~ X1 + X2 | id_indiv + id_firm + id_year"

    # warmup
    df_warmup, _, _, _, _ = generate_complex_fixed_effects_data(n=10**4)
    pf.feols(
        fml,
        data=df_warmup,
        demeaner_backend=demeaner_backend,
        store_data=False,
        copy_data=False,
        solver=solver,
        fixef_tol=1e-06,
        fixef_maxiter=500_000,
    )

    for rep in range(nrep):
        df, Y, X, f, weights = generate_complex_fixed_effects_data(n=n)

        tic1 = time.time()
        pf.feols(
            fml,
            data=df,
            demeaner_backend=demeaner_backend,
            store_data=False,
            copy_data=False,
            solver=solver,
            fixef_tol=1e-06,
            fixef_maxiter=500_000,
        )
        tic2 = time.time()

        full_feols_timing = tic2 - tic1

        res.append(
            pd.Series(
                {
                    "method": "feols",
                    "solver": solver,
                    "demeaner_backend": demeaner_backend,
                    "n_obs": df.shape[0],
                    "k": 2,  # Always 2 covariates
                    "G": 3,  # Always 3 fixed effects
                    "rep": rep + 1,
                    "full_feols_timing": full_feols_timing,
                    "demean_timing": np.nan,
                }
            )
        )

    return pd.concat(res, axis=1).T


a_rust = run_standard_benchmark(
    fixed_effect="dum_1", demeaner_backend="rust", size=1, k=1
)
a_cupy = run_standard_benchmark(
    fixed_effect="dum_1", demeaner_backend="cupy", size=1, k=1
)
print(a_rust)
print(a_cupy)


def run_all_benchmarks(size_list, k_list, nrep):
    """
    Run all the benchmarks.

    Args:
        size_list (list): The list of sizes to run the benchmarks on. 1-> 1000, 2-> 10000, ..., 5-> 10_000_000
        k_list (list): The list of k values to run the benchmarks on.
    """
    res = pd.DataFrame()

    all_combinations = list(
        product(
            [
                "numba",
                "rust",
                "cupy64",
                "cupy32",
                #    "scipy"
            ],  # demeaner_backend
            ["dum_1", "dum_1+dum_2", "dum_1+dum_2+dum_3"],  # fixef
            size_list,  # size
            k_list,  # k
            ["np.linalg.solve"],  # solver
        )
    )

    with tqdm(total=len(all_combinations), desc="Running Standard Benchmarks") as pbar:
        for demeaner_backend, fixef, size, k, solver in all_combinations:
            res = pd.concat(
                [
                    res,
                    run_standard_benchmark(
                        solver=solver,
                        fixed_effect=fixef,
                        demeaner_backend=demeaner_backend,
                        size=size,
                        k=k,
                        nrep=nrep,
                    ),
                ],
                axis=0,
            )
            pbar.update(1)  # Update the progress bar after each iteration

    return res


def run_all_complex_benchmarks(n_list, nrep):
    """
    Run all the complex benchmarks.

    Complex benchmarks always use:
    - 3 fixed effects: id_indiv + id_firm + id_year
    - 2 covariates: X1 + X2

    Args:
        n_list (list): The list of n (observation counts) to run the benchmarks on.
        nrep (int): Number of repetitions for each benchmark.
    """
    res = pd.DataFrame()

    all_combinations = list(
        product(
            ["numba", "rust", "cupy64", "cupy32", "scipy"],  # demeaner_backend
            n_list,  # n
            ["np.linalg.solve"],  # solver
        )
    )

    with tqdm(total=len(all_combinations), desc="Running Complex Benchmarks") as pbar:
        for demeaner_backend, n, solver in all_combinations:
            res = pd.concat(
                [
                    res,
                    run_complex_benchmark(
                        solver=solver, demeaner_backend=demeaner_backend, n=n, nrep=nrep
                    ),
                ],
                axis=0,
            )
            pbar.update(1)  # Update the progress bar after each iteration

    return res


# ============================================================================
# STANDARD BENCHMARKS
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING STANDARD BENCHMARKS")
print("=" * 80 + "\n")

res_all_standard = run_all_benchmarks(
    size_list=[1, 2, 3, 4, 5],  # for N = 1000, 10_000, 100_000, 1_000_000, 10_000_000
    k_list=[1, 10, 25],  # for k = 1, 10, 25
    nrep=3,
)

df_standard = (
    res_all_standard.drop(["rep", "solver"], axis=1)
    .groupby(["method", "demeaner_backend", "k", "G", "n_obs"])
    .mean()
    .reset_index()
)

print("\nStandard Benchmark Results:")
print(df_standard)

df_standard.to_csv("gpu_runtime_res_standard.csv", index=False)
print("\nSaved standard results to: gpu_runtime_res_standard.csv")

# Create standard benchmark plot
g_standard = create_benchmark_plot(
    df_standard, "Standard Data", "gpu_runtime_comparison_standard.png"
)

# ============================================================================
# COMPLEX BENCHMARKS
# ============================================================================
print("\n" + "=" * 80)
print("RUNNING COMPLEX BENCHMARKS")
print("=" * 80 + "\n")

res_all_complex = run_all_complex_benchmarks(
    n_list=[10**5, 10**6, 5 * 10**6, 10**7],  # Various observation counts
    nrep=3,
)

df_complex = (
    res_all_complex.drop(["rep", "solver"], axis=1)
    .groupby(["method", "demeaner_backend", "k", "G", "n_obs"])
    .mean()
    .reset_index()
)

print("\nComplex Benchmark Results:")
print(df_complex)

df_complex.to_csv("gpu_runtime_res_complex.csv", index=False)
print("\nSaved complex results to: gpu_runtime_res_complex.csv")

# Create complex benchmark plot (single plot, no faceting)
g_complex = create_benchmark_plot(
    df_complex,
    "Complex Data",
    "gpu_runtime_comparison_complex.png",
    facet_by_fixef=False,
)

# Show plots
plt.show()

print("\n" + "=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)
