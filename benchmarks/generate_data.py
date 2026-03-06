#!/usr/bin/env python
"""Generate benchmark datasets and save to parquet for use by all languages.

This script creates reproducible datasets that R, Python, and Julia all use.
"""

import json
from pathlib import Path

import numpy as np
from dgp_functions import base_dgp

# Project root
ROOT = Path(__file__).resolve().parent

# Create data directory if needed
(data_dir := ROOT / "data").mkdir(exist_ok=True)

# Configuration
config_path = Path("config.json")
config = json.loads(config_path.read_text()) if config_path.exists() else {}

# Datasets to generate
datasets = config.get(
    "datasets",
    [
        {"name": "simple_1k", "n": int(1e3), "type": "simple"},
        {"name": "difficult_1k", "n": int(1e3), "type": "difficult"},
        {"name": "simple_10k", "n": int(1e4), "type": "simple"},
        {"name": "difficult_10k", "n": int(1e4), "type": "difficult"},
        {"name": "simple_100k", "n": int(1e5), "type": "simple"},
        {"name": "difficult_100k", "n": int(1e5), "type": "difficult"},
        {"name": "simple_500k", "n": int(5e5), "type": "simple"},
        {"name": "difficult_500k", "n": int(5e5), "type": "difficult"},
        {"name": "simple_1m", "n": int(1e6), "type": "simple"},
        {"name": "difficult_1m", "n": int(1e6), "type": "difficult"},
        {"name": "simple_2m", "n": int(2e6), "type": "simple"},
        {"name": "difficult_2m", "n": int(2e6), "type": "difficult"},
    ],
)

# Number of iterations per dataset (including burn-in)
iterations = config.get("iterations", {})
n_iters = int(iterations.get("n_iters", 3))
burn_in = int(iterations.get("burn_in", 1))

np.random.seed(20250725)

print("=" * 80)
print("GENERATING BENCHMARK DATASETS")
print("=" * 80)
print(f"Output directory: {data_dir}")
print(f"Iterations per dataset: {n_iters} (+ {burn_in} burn-in)")
print()

for ds in datasets:
    ds_name = ds["name"]
    ds_n = int(ds["n"])
    ds_type = ds["type"]
    print(f"Dataset: {ds_name:<20s} (n={ds_n:,}, type={ds_type})")

    for iter_ in range(1, n_iters + burn_in + 1):
        if iter_ <= burn_in:
            iter_type = "burnin"
            iter_num = iter_
        else:
            iter_type = "iter"
            iter_num = iter_ - burn_in

        filename = f"{ds_name}_{iter_type}_{iter_num}.parquet"
        filepath = data_dir / filename

        print(f"  -> {filename} ... ", end="", flush=True)

        df = base_dgp(n=ds_n, type_=ds_type)
        df.to_parquet(filepath)

        print(f"done ({len(df):,} rows)")

    print()

total_files = len(datasets) * (n_iters + burn_in)
print("=" * 80)
print("DATA GENERATION COMPLETE")
print("=" * 80)
print(f"Total files: {total_files}")
print(f"Location: {data_dir}")
