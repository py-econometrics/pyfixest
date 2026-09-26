# Multiple-estimation benchmarks

`benchmark_multiple.py` compares a complete multiple-estimation call with the
same regressions fitted independently. Both paths retain their results until
the clock stops. It checks named coefficients, full covariance matrices,
standard errors, and observation counts outside the timed section.

Run from a checkout using Pixi. The R reference also needs the `py312-r`
environment. Warm imports and build the extension before benchmarking; confirm
the recorded checkout and extension paths belong to the intended build.

```sh
export PYTHONPATH=.
pixi run -e py312 python benchmarks/modular/benchmark_multiple.py prepare /tmp/multi-data \
  --dgps simple difficult --sizes 1000 10000 100000 1000000 \
  --seeds 20260926 20260927 20260928
pixi run -e py312 python benchmarks/modular/benchmark_multiple.py run /tmp/multi-data \
  --output /tmp/multi-python --backends map within --reps 5
pixi run -e py312-r python benchmarks/modular/benchmark_multiple.py run /tmp/multi-data \
  --output /tmp/multi-r --backends fixest --reps 5
pixi run -e py312 python benchmarks/modular/benchmark_multiple.py summarize \
  /tmp/multi-python /tmp/multi-r --output /tmp/multi-summary.csv
```

For a short smoke run, prepare `--sizes 1000 --models 3 --controls 2
--fe-counts 2` and run `--reps 2`. Preparation and execution refuse to overwrite
existing manifests or result files. Use a fresh directory for another run.
Datasets are CSV for identical Python/R inputs; SHA-256 hashes detect changes.
The manifest records seeds, formulas, realized rows, FE counts, and AKM mobility
and connectivity diagnostics. All seeds are explicit and stable across processes.

## Benchmark matrix

| Dimension | Ordinary comparison | Targeted extension |
|---|---|---|
| DGP | `simple difficult` | `akm akm_low_mobility` (mobility 0.2/0.05) |
| Rows | 1k, 10k, 100k, 1m | 10m when memory permits |
| FE | worker + firm; worker + firm + year | `--fe-counts 0 1` controls |
| Pattern | `csw shared lhs sw` | filter with `--patterns` / `--match` |
| Models | 10 | `--models 2` and `--models 50` |
| Shared controls | 10 | `--controls 1` and `--controls 50` |
| Sample | clean | `--scenario weights`, `missing`, or `singletons` |
| Storage | ordinary retained fits | `--lean` or `--no-store-data` |
| Backend | Rust MAP, within with additive preconditioner | R fixest reference |

`csw` adds regressors cumulatively; `shared` keeps common controls and switches
one regressor; `lhs` varies outcomes with identical X; `sw` switches disjoint
regressors. The difficult base DGP differs in firm assignment, so firm FE must
be present to exercise that difference. Firm and worker counts scale with N;
this is not a fixed-FE-count experiment. AKM data use ten periods per worker.
The outcome is the existing DGP outcome with independent candidate regressors;
additional outcomes add independent noise to it.

The ordinary matrix uses IID inference, no observation weights, no singleton
removal, tolerance 1e-8, and 10,000 demeaning iterations. The targeted weights
scenario uses analytic weights. The singleton scenario enables singleton
removal; missingness is model-specific. All these settings match across
Python and R. Retention internals differ across packages even with analogous
flags; use within-package comparisons to evaluate an implementation change.

## Measurement and interpretation

Each case/backend runs sequentially in its own subprocess. One untimed fit of
each path warms the runtime and validates results, followed by alternating
independent/multiple measurements. Each call starts with an empty estimation
cache. Dataset loading, correctness checks, garbage collection, and destruction
of results happen outside the timer. Thread environment variables are set
before worker imports; R's thread count is explicit. Start with `--threads 1`,
then repeat a selected workload with a fixed larger count. Do not benchmark
while tests, compilers, or other CPU-intensive jobs are running.

A predeclared `--timeout` (default 300 seconds per case/backend, including
startup and warmup) records `timeout` instead of silently omitting a hard case.
Errors have `failed` status and a companion log. Failed/timed-out cases remain
in the summary; never change tolerances or iterations to manufacture a win.
The timeout is a process wall-clock limit, not a per-fit limit.

JSON preserves every timing, estimates, environment, settings, Git SHA, tracked
diff hash, native-extension hash, and dataset hash. CSV reports median/min/max
seconds and independent/multiple speedup. Python also reports process peak RSS
(in bytes, including imports, data, correctness warmup, and both fit paths).
This is a process high-water mark, not an isolated estimate of cache memory or
per-path memory. Native allocations are included; R RSS is not collected.

For base-versus-PR comparisons, use separate worktrees with the **same saved
manifest** and harness version. The harness can be executed by absolute path
with `PYTHONPATH=<target-checkout>:<harness-checkout>` using the target's Pixi
environment. Inspect `environment.checkout` and `head` before interpreting a
run. Rebuild after native changes. Preserve logs and result JSON with the PR's
benchmark evidence; do not commit generated datasets.

Report all four medians: base independent, base multiple, PR independent, PR
multiple. Report `base_multiple / PR_multiple` separately from each version's
independent/multiple ratio. Alternate the order of base/PR runs, repeat across
three saved seeds, and inspect variability. A speedup ratio can improve merely
because independent fits got slower. Require a repeatable full-call improvement
on the targeted pattern, no unexplained control regression, and acceptable RSS.
R is a reference for opportunities and correctness, not a universal 10x target.

Run diagnostic instrumentation separately from timing:

```sh
pixi run -e py312 python benchmarks/modular/benchmark_multiple.py run /tmp/multi-data \
  --output /tmp/multi-profile --match simple-100000- --profile
```

This records columns sent to each demeaner call, preconditioner reuse, and a
cProfile `.prof` file for one warmed multiple fit. Profile results have no timing
speedup and must not be mixed into performance claims. Inspect model-matrix,
cache selection, projection, and fitting phases with `pstats`. For ten cumulative
single-regressor additions on one sample, expect backend column counts
`[2, 1, ..., 1]`; splitting samples should produce distinct cache populations.
