# Demeaning Benchmarks

Benchmarks for comparing demeaning backends on worker-firm-year panel data with tunable graph structure. The underlying DGP generates three-way fixed effects panels where difficulty is controlled by how connected the worker-firm bipartite graph is — sparser, more segmented graphs require more MAP iterations to converge.

## What makes demeaning hard?

The MAP (Method of Alternating Projections) algorithm converges faster when the bipartite graph of workers and firms is well-connected. The key difficulty levers are:

- **Mobility** (`p_move`): The probability a worker switches firms each year. Low mobility means fewer cross-firm links, producing a sparse graph that converges slowly.
- **Cluster structure** (`n_clusters`, `p_between_cluster`): Firms are grouped into clusters. When cross-cluster moves are rare, the graph decomposes into near-disconnected components — the hardest case for iterative demeaning.
- **Firm size skew** (`pareto_shape`): Lower values produce a few mega-firms and many tiny firms. Extreme skew means most firms contribute little connectivity.
- **Scale** (`n_workers`, `n_firms`): Larger matrices cost more per iteration, independent of convergence difficulty.
- **Panel unbalancedness** (`p_observe`, `spell_concentration`): Missing observations remove edges from the graph. Combined with selection (`selection_worker`), this can thin out connectivity further.
- **Firm survival** (`p_survive`, `selection_firm`): Firm death and entry reshuffles the graph over time.
- **Sorting** (`sorting_wf`): Positive assortative matching (high-type workers at high-type firms) doesn't directly affect convergence but changes the data structure.
- **Number of features** (`n_features`): More columns to demean simultaneously tests vectorization efficiency. Does not affect convergence (same graph, same iterations).

## Scenarios

Four scenarios that combine these levers at increasing difficulty, from a quick sanity check to a stress test:

| Scenario | Workers | Firms | Years | What makes it easy/hard |
|----------|---------|-------|-------|-------------------------|
| **easy** | 10k | 1k | 10 | High mobility (15%), no clusters, balanced panel, no firm exit. A single well-connected component — converges in few iterations. |
| **medium** | 50k | 5k | 15 | Lower mobility (5%), 5 clusters with 30% cross-cluster moves, 80% observation rate, some firm exit. Moderate graph segmentation. |
| **hard** | 100k | 10k | 20 | Low mobility (2%), 10 clusters with 10% cross-cluster moves, 60% observation rate, heavy firm exit, worker-firm sorting, extreme firm size skew (Pareto shape 1.0). A near-disconnected, sparse, unbalanced graph. |
| **large** | 150k | 15k | 15 | Same convergence difficulty as hard (2% mobility, 10 clusters, 10% cross-cluster) but at 1.5x scale. Tests whether your algorithm handles a near-disconnected graph with 150k workers. |

## Parameter sweeps

Each sweep varies a single parameter while holding everything else at the `large` scenario defaults. This isolates the effect of each difficulty lever.

| Sweep | Parameter | Values | What it tests |
|-------|-----------|--------|---------------|
| **mobility** | `p_move` | 0.01, 0.10, 0.30 | Direct control over graph connectivity. The strongest driver of convergence speed — expect ~35x timing spread across values. |
| **pareto** | `pareto_shape` | 1.0, 3.0, 10.0 | Firm size distribution skew. Shape 1.0 = extreme skew (a few huge firms), 10.0 = nearly uniform sizes. Most effect at the low end. |
| **cluster** | `n_clusters` / `p_between_cluster` | (1, 1.0), (5, 0.1), (20, 0.05) | Graph segmentation. (1, 1.0) = no clusters; (20, 0.05) = 20 near-isolated labor markets. Tests whether the algorithm struggles with block-diagonal structure. |
| **group_count** | `n_workers` / `n_firms` | 3k/300, 30k/3k, 150k/15k | Pure scaling test with a fully balanced, fully connected panel (no missingness, no firm exit). Isolates per-iteration cost from convergence difficulty. |
| **features** | `n_features` | 1, 5, 20 | Number of columns demeaned simultaneously. Same DGP, same iterations — tests whether the backend vectorizes across features efficiently. |

## Step 1: Baseline the existing backends

```bash
# Run scenarios (easy/medium/hard/large)
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
--scenarios           Run main scenarios (easy/medium/hard/large)
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
