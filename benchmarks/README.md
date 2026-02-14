# Demeaning Benchmarks

## Step 1: Baseline the existing backends

```bash
# Run scenarios (easy/medium/hard/extreme)
pixi run -e dev python -m benchmarks.run_benchmarks --scenarios --backends numba rust scipy --reps 3 --save-baseline v1

# Run parameter sweeps
pixi run -e dev python -m benchmarks.run_benchmarks --sweeps --backends numba rust scipy --reps 3 --save-baseline v1-sweeps
```

Results are in `benchmarks/results/`. Baselines are saved to `benchmarks/results/baselines/`.

## Step 2: Implement your algorithm

Register your backend in `pyfixest/estimation/backends.py`. Your `demean` function can return either:

- `(result, converged)` — standard 2-tuple
- `(result, converged, n_iterations)` — 3-tuple, enables iteration tracking

## Step 3: Compare against the baseline

```bash
# Scenarios
pixi run -e dev python -m benchmarks.run_benchmarks --scenarios --backends numba rust scipy mybackend --reps 3 --baseline v1

# Sweeps
pixi run -e dev python -m benchmarks.run_benchmarks --sweeps --backends numba rust scipy mybackend --reps 3 --baseline v1-sweeps
```

This prints a comparison table with speedup, delta %, and geometric mean. Your new backend appears as "new". Output is also saved to `benchmarks/results/baseline_comparison_*.txt` and `*.png`.

You can save and compare in one step by combining both flags:

```bash
pixi run -e dev python -m benchmarks.run_benchmarks --scenarios --backends numba rust scipy mybackend --reps 3 --save-baseline v2 --baseline v1
```

## CLI Reference

```
--scenarios           Run main scenarios (easy/medium/hard/extreme)
--sweeps              Run parameter sweeps
--all                 Run everything (default)
--backends B [B ...]  Backends to benchmark (default: numba)
--reps N              Repetitions per scenario (default: 3)
--save-baseline NAME  Save results as a named baseline
--baseline NAME       Compare against a saved baseline
--sweep-name NAME     Run a single sweep (e.g. mobility, features, group_count)
--n-features N        Columns to demean (default: 1)
--fe-columns COL ...  FE columns (default: worker_id firm_id year)
--feols               Also benchmark feols() (slower)
```
