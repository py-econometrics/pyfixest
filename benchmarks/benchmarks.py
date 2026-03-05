#!/usr/bin/env python3
"""
Python benchmark runner for fixed-effect estimation.
Runs in a single persistent process so numba JIT compilation persists across iterations.
Uses multiprocessing for proper timeout of slow estimators (statsmodels, linearmodels).
"""

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Optional JAX availability detection
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Optional torch availability detection
try:
    import torch

    HAS_TORCH = True
    HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False
    HAS_CUDA = False

# Backends that accept a backend= argument when called through pyfixest runners
_PYFIXEST_BACKENDS = {
    "scipy",
    "numba",
    "rust",
    "jax",
    "torch_cpu",
    "torch_mps",
    "torch_cuda",
    "torch_cuda32",
}

# =============================================================================
# Helpers
# =============================================================================


def _append_optional_backends(estimators, label_prefix, runner_func, func_name):
    """Append JAX + torch backend estimators based on runtime availability."""
    optional = []
    if HAS_JAX:
        optional.append(("jax", "jax"))
    if HAS_TORCH:
        optional.append(("torch_cpu", "torch_cpu"))
    if HAS_MPS:
        optional.append(("torch_mps", "torch_mps"))
    if HAS_CUDA:
        optional.append(("torch_cuda", "torch_cuda"))
        optional.append(("torch_cuda32", "torch_cuda32"))
    for suffix, backend in optional:
        estimators.append(
            (f"{label_prefix} ({suffix})", backend, runner_func, False, func_name)
        )


# =============================================================================
# Estimator functions (run in main process for JIT caching)
# =============================================================================


def run_pyfixest_feols(data: pd.DataFrame, formula: str, backend: str) -> float:
    """Run pyfixest feols and return timing."""
    import pyfixest as pf

    start = time.perf_counter()
    _ = pf.feols(formula, data, demeaner_backend=backend)
    return time.perf_counter() - start


def run_pyfixest_fepois(data: pd.DataFrame, formula: str, backend: str) -> float:
    """Run pyfixest fepois and return timing."""
    import pyfixest as pf

    start = time.perf_counter()
    _ = pf.fepois(formula, data, demeaner_backend=backend)
    return time.perf_counter() - start


def run_pyfixest_feglm_logit(data: pd.DataFrame, formula: str, backend: str) -> float:
    """Run pyfixest feglm (logit) and return timing."""
    import pyfixest as pf

    start = time.perf_counter()
    _ = pf.feglm(formula, data, family="logit", demeaner_backend=backend)
    return time.perf_counter() - start


def parse_fe_formula(formula: str) -> tuple[str, list[str]]:
    """Parse formula like 'y ~ x1 | fe1 + fe2' into main formula and FE list."""
    if "|" in formula:
        main_part, fe_part = formula.split("|", 1)
        main_formula = main_part.strip()
        fe_names = [fe.strip() for fe in fe_part.split("+")]
        return main_formula, fe_names
    return formula, []


def run_absorbingls(data: pd.DataFrame, formula: str) -> float:
    """Run linearmodels AbsorbingLS and return timing."""
    from formulaic import model_matrix
    from linearmodels.iv.absorbing import AbsorbingLS

    main_formula, fe_names = parse_fe_formula(formula)

    start = time.perf_counter()
    y, X = model_matrix(main_formula, data)
    absorb = data[fe_names].astype("category") if fe_names else None
    mod = AbsorbingLS(y, X, absorb=absorb)
    _ = mod.fit()
    return time.perf_counter() - start


def run_statsmodels_ols(data: pd.DataFrame, formula: str) -> float:
    """Run statsmodels OLS (with categorical FEs as dummies)."""
    import statsmodels.formula.api as smf

    main_formula, fe_names = parse_fe_formula(formula)

    if fe_names:
        fe_terms = " + ".join(f"C({fe})" for fe in fe_names)
        full_formula = f"{main_formula} + {fe_terms}"
    else:
        full_formula = main_formula

    start = time.perf_counter()
    mod = smf.ols(full_formula, data=data)
    _ = mod.fit()
    return time.perf_counter() - start


# =============================================================================
# Subprocess wrapper for timeout support
# =============================================================================


def _run_in_subprocess(
    func_name: str,
    data_path: str,
    formula: str,
    result_queue: mp.Queue,
    backend: str | None = None,
):
    """Worker function that runs in subprocess."""
    try:
        data = pd.read_parquet(data_path)
        if func_name == "absorbingls":
            elapsed = run_absorbingls(data, formula)
        elif func_name == "statsmodels_ols":
            elapsed = run_statsmodels_ols(data, formula)
        elif func_name == "pyfixest_feols":
            elapsed = run_pyfixest_feols(data, formula, backend)
        elif func_name == "pyfixest_fepois":
            elapsed = run_pyfixest_fepois(data, formula, backend)
        elif func_name == "pyfixest_feglm_logit":
            elapsed = run_pyfixest_feglm_logit(data, formula, backend)
        else:
            raise ValueError(f"Unknown function: {func_name}")  # noqa: TRY301
        result_queue.put(("success", elapsed))
    except MemoryError:
        result_queue.put(("oom", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def run_with_timeout(
    func_name: str,
    data_path: str,
    formula: str,
    timeout: int,
    backend: str | None = None,
) -> tuple[str, float | None]:
    """Run a function in subprocess with timeout. Returns (status, elapsed_time)."""
    result_queue = mp.Queue()
    proc = mp.Process(
        target=_run_in_subprocess,
        args=(func_name, data_path, formula, result_queue, backend),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return ("timeout", None)

    if result_queue.empty():
        return ("error", None)

    return result_queue.get()


# =============================================================================
# Benchmark configuration
# =============================================================================


def load_config(config_path: str | Path) -> dict:
    """Load a JSON benchmark config from the given path."""
    if not Path(config_path).exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def get_formulas_from_config(
    config: dict, benchmark_type: str
) -> dict[int, str] | None:
    """Obtain regression formulas from an already loaded config."""
    formulas_cfg = config.get("formulas", {}).get(benchmark_type, {})
    if not formulas_cfg:
        return None
    formulas = {}
    for n_fe_str, entry in formulas_cfg.items():
        if "python" in entry:
            formulas[int(n_fe_str)] = entry["python"]
    return formulas or None


def get_allowed_datasets(config: dict, benchmark_type: str) -> set[str] | None:
    """Get the allowed datasets from an already loaded config."""
    allowed = config.get("datasets_by_type", {}).get(benchmark_type)
    if not allowed:
        return None
    return set(allowed)


def get_estimators(
    benchmark_type: str, timeout_estimators: set[str]
) -> tuple[list, dict[int, str]]:
    """Get estimators and formulas for benchmark type.

    Returns: list of (name, backend/func_str, func, use_subprocess, func_name_for_subprocess)
    """
    if benchmark_type == "ols":
        estimators = [
            (
                "pyfixest.feols (scipy)",
                "scipy",
                run_pyfixest_feols,
                False,
                "pyfixest_feols",
            ),
            (
                "pyfixest.feols (numba)",
                "numba",
                run_pyfixest_feols,
                False,
                "pyfixest_feols",
            ),
            (
                "pyfixest.feols (rust)",
                "rust",
                run_pyfixest_feols,
                False,
                "pyfixest_feols",
            ),
        ]
        _append_optional_backends(
            estimators, "pyfixest.feols", run_pyfixest_feols, "pyfixest_feols"
        )
        estimators += [
            (
                "linearmodels.AbsorbingLS",
                "absorbingls",
                run_absorbingls,
                "linearmodels.AbsorbingLS" in timeout_estimators,
                "absorbingls",
            ),
            (
                "statsmodels.OLS",
                "statsmodels_ols",
                run_statsmodels_ols,
                "statsmodels.OLS" in timeout_estimators,
                "statsmodels_ols",
            ),
        ]
        formulas = {
            2: "y ~ x1 | indiv_id + year",
            3: "y ~ x1 | indiv_id + year + firm_id",
        }
    elif benchmark_type == "poisson":
        estimators = [
            (
                "pyfixest.fepois (scipy)",
                "scipy",
                run_pyfixest_fepois,
                False,
                "pyfixest_fepois",
            ),
            (
                "pyfixest.fepois (numba)",
                "numba",
                run_pyfixest_fepois,
                False,
                "pyfixest_fepois",
            ),
            (
                "pyfixest.fepois (rust)",
                "rust",
                run_pyfixest_fepois,
                False,
                "pyfixest_fepois",
            ),
        ]
        _append_optional_backends(
            estimators, "pyfixest.fepois", run_pyfixest_fepois, "pyfixest_fepois"
        )
        formulas = {
            2: "negbin_y ~ x1 | indiv_id + year",
            3: "negbin_y ~ x1 | indiv_id + year + firm_id",
        }
    elif benchmark_type == "logit":
        estimators = [
            (
                "pyfixest.feglm_logit (scipy)",
                "scipy",
                run_pyfixest_feglm_logit,
                False,
                "pyfixest_feglm_logit",
            ),
            (
                "pyfixest.feglm_logit (numba)",
                "numba",
                run_pyfixest_feglm_logit,
                False,
                "pyfixest_feglm_logit",
            ),
            (
                "pyfixest.feglm_logit (rust)",
                "rust",
                run_pyfixest_feglm_logit,
                False,
                "pyfixest_feglm_logit",
            ),
        ]
        _append_optional_backends(
            estimators,
            "pyfixest.feglm_logit",
            run_pyfixest_feglm_logit,
            "pyfixest_feglm_logit",
        )
        formulas = {
            2: "binary_y ~ x1 | indiv_id + year",
            3: "binary_y ~ x1 | indiv_id + year + firm_id",
        }
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    return estimators, formulas


def parse_dataset_name(name: str) -> tuple[str, int]:
    """Parse dataset name like 'simple_1k' into (type, n_obs)."""
    size_map = {
        "1k": 1_000,
        "10k": 10_000,
        "100k": 100_000,
        "500k": 500_000,
        "1m": 1_000_000,
        "2m": 2_000_000,
        "5m": 5_000_000,
    }
    parts = name.rsplit("_", 1)
    dgp_type = parts[0]
    n_str = parts[1] if len(parts) > 1 else "unknown"
    n_obs = size_map.get(n_str, 0)
    return dgp_type, n_obs


# =============================================================================
# Main benchmark runner
# =============================================================================


def run_benchmark(
    data_dir: Path,
    output_file: Path,
    benchmark_type: str,
    timeout: int = 60,
    filter_pattern: str | None = None,
    timeout_estimators: set[str] | None = None,
    formulas_override: dict[int, str] | None = None,
    allowed_datasets: set[str] | None = None,
) -> None:
    """Run benchmarks on all datasets in data_dir."""
    if timeout_estimators is None:
        timeout_estimators = set()
    estimators, formulas = get_estimators(benchmark_type, timeout_estimators)
    if formulas_override:
        formulas = formulas_override

    # Get all parquet files
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Group files by dataset (excluding iteration suffix)
    datasets = defaultdict(list)
    for f in parquet_files:
        # Parse filename: simple_1k_burnin_1.parquet or simple_1k_iter_1.parquet
        parts = f.stem.rsplit("_", 2)
        if len(parts) >= 3:
            ds_name = parts[0]
            iter_type = parts[1]
            iter_num = int(parts[2])
            # Apply filter if specified
            if filter_pattern and filter_pattern not in ds_name:
                continue
            if allowed_datasets is not None and ds_name not in allowed_datasets:
                continue
            datasets[ds_name].append((iter_type, iter_num, f))

    results = []

    print("\n" + "=" * 80)
    print(f"PYTHON BENCHMARK: {benchmark_type.upper()}")
    print("=" * 80)
    filter_info = f" | Filter: '{filter_pattern}'" if filter_pattern else ""
    print(
        f"Estimators: {len(estimators)} | FE configs: {len(formulas)} | Timeout: {timeout}s{filter_info}"
    )

    for ds_name, files in sorted(datasets.items()):
        dgp_type, n_obs = parse_dataset_name(ds_name)

        print(f"\n{'-' * 80}")
        print(f"Dataset: {ds_name} (n={n_obs:,})")
        print(f"{'-' * 80}")

        # Sort files: burnin first, then iter
        files_sorted = sorted(files, key=lambda x: (0 if x[0] == "burnin" else 1, x[1]))

        for iter_type, iter_num, filepath in files_sorted:
            print(f"\n[{iter_type} {iter_num}] Loading {filepath.name}...")
            data = pd.read_parquet(filepath)

            for n_fe, formula in formulas.items():
                for (
                    est_name,
                    backend_or_func,
                    func,
                    use_subprocess,
                    func_name_subprocess,
                ) in estimators:
                    print(f"  -> {est_name:<35} (FE={n_fe}) ... ", end="", flush=True)

                    try:
                        if use_subprocess:
                            # Run in subprocess with timeout (statsmodels only)
                            status, elapsed = run_with_timeout(
                                func_name_subprocess,
                                str(filepath),
                                formula,
                                timeout,
                                None,
                            )
                            if status == "timeout":
                                print("TIMEOUT")
                                elapsed = None
                            elif status == "oom":
                                print("OOM")
                                elapsed = None
                            elif status == "error":
                                print(f"ERROR: {elapsed}")
                                elapsed = None
                            else:
                                print(f"{elapsed:.3f}s")
                        else:
                            # Run in main process
                            if backend_or_func in _PYFIXEST_BACKENDS:
                                elapsed = func(data, formula, backend_or_func)
                            else:
                                elapsed = func(data, formula)
                            print(f"{elapsed:.3f}s")

                        # Only record non-burnin iterations
                        if iter_type != "burnin":
                            results.append(
                                {
                                    "iter": iter_num,
                                    "time": elapsed,
                                    "est_name": est_name,
                                    "n_fe": n_fe,
                                    "dgp_name": dgp_type,
                                    "n_obs": n_obs,
                                }
                            )

                    except MemoryError:
                        print("OOM")
                        if iter_type != "burnin":
                            results.append(
                                {
                                    "iter": iter_num,
                                    "time": None,
                                    "est_name": est_name,
                                    "n_fe": n_fe,
                                    "dgp_name": dgp_type,
                                    "n_obs": n_obs,
                                }
                            )

                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(
                            x in error_msg
                            for x in ["svd", "singular", "convergence", "demeaning"]
                        ):
                            print("NUMERICAL_ERROR")
                        else:
                            print(f"ERROR: {e}")
                        if iter_type != "burnin":
                            results.append(
                                {
                                    "iter": iter_num,
                                    "time": None,
                                    "est_name": est_name,
                                    "n_fe": n_fe,
                                    "dgp_name": dgp_type,
                                    "n_obs": n_obs,
                                }
                            )

    # Write results to CSV
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["iter", "time", "est_name", "n_fe", "dgp_name", "n_obs"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_file}")


def main():
    """Run the main benchmark loop."""
    HERE = Path(__file__).parent
    # Required for multiprocessing on macOS
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Run Python fixed-effect benchmarks")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE / "data",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / "bench_python.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--type",
        choices=["ols", "poisson", "logit"],
        default="ols",
        help="Benchmark type",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per estimation in seconds (default: 60)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter datasets by name (e.g., 'simple' to exclude 'difficult')",
    )
    args = parser.parse_args()

    config = load_config("bench.json")
    timeout_secs = config.get("timeout_secs", {}).get("python", args.timeout)
    timeout_estimators = set(config.get("python_timeout_estimators", []))
    if not timeout_estimators:
        timeout_estimators = {"linearmodels.AbsorbingLS", "statsmodels.OLS"}
    formulas_override = get_formulas_from_config(config, args.type)
    allowed_datasets = get_allowed_datasets(config, args.type)

    run_benchmark(
        args.data_dir,
        args.output,
        args.type,
        timeout_secs,
        args.filter,
        timeout_estimators,
        formulas_override,
        allowed_datasets,
    )


if __name__ == "__main__":
    main()
