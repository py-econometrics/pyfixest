# Testing and verification

Pyfixest testing is risk-based: run the narrowest useful checks while editing,
then the affected long-running suites after the change stabilizes. Always use
`pixi run`; bare Python or pytest may not have the compiled extension or the
right optional dependencies.

## Runtime tiers

Runtime labels are qualitative because machine, compiler cache, and test
selection affect wall time.

| Tier | Purpose | Typical scale | Examples |
|---|---|---|---|
| Edit | Fast feedback on the changed seam | Seconds to a few minutes | targeted pytest, changed-file Ruff/mypy |
| PR baseline | Broad Python regression check | Minutes | `pixi run test-py` |
| Domain/special | Validate an affected environment or reference | May take tens of minutes | R, HAC, no-JIT, docs, plots, Rust |
| Exhaustive | Release/platform confidence | Potentially substantially longer | `test-all`, platform CI, benchmarks |

Run edit checks repeatedly. Run required domain checks once the design is
stable. A verification report records actual elapsed time and reports every
applicable check as passed, failed, deferred, or not run. Deferred is never
equivalent to passed.

## Deterministic change verification

`scripts/agent/verification_matrix.toml` is the machine-readable source of
check commands, domains, runtime tiers, and local/CI requirements.
`change_scope.py` compares the worktree with a verified base and classifies
the affected domains and risk flags. `verify.py` uses that scope to select and
run checks without shell interpolation.

```bash
# Inspect paths, domains, and risks
pixi run agent-scope --base <immediate-parent>

# Preview the PR-tier commands
pixi run agent-verify --base <immediate-parent> --tier pr --dry-run

# Run them and optionally write a versioned JSON report
pixi run agent-verify --base <immediate-parent> --tier pr \
  --json-output /tmp/pyfixest-verification.json
```

For a stack, verify each layer against its immediate parent and verify the top
layer cumulatively against `master`. The available tiers are `edit`, `pr`,
`domain`, and `exhaustive`.

A long check may be declared as CI work with
`--defer CHECK_ID=REASON` only when its matrix entry allows CI deferral.
Deferring a required local check produces a failing exit code. Invalid
configuration and Git-scope errors exit with status 2.

## Commands

```bash
# Targeted, without the repository-wide coverage report
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov

# Python baseline
pixi run test-py

# External references
pixi run test-r-core
pixi run test-r-fixest
pixi run test-r-hac
pixi run test-r-extended

# Exhaustive
pixi run test-all

# Changed-file quality checks
pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check --files <changed files>
pixi run -e lint prek run mypy --files <changed files>

# Documentation
pixi run docs-build
pixi run docs-render
```

## Selection matrix

| Change | Required local evidence | Long or CI evidence |
|---|---|---|
| Documentation only | formatting/link checks; execute changed examples | `docs-build` and `docs-render` when applicable |
| Python API or internals | targeted public tests; changed-file lint/type checks | Python baseline |
| Estimation or inference numerics | targeted integration and edge tests | applicable live external-reference suite |
| New estimator | complete support matrix and permanent external comparison | Python baseline, external suite, full platform CI |
| HAC | targeted HAC/meat tests | single-threaded `test-r-hac` |
| Rust | kernel/reference integration tests | Python baseline, platform CI, and relevant benchmarks |
| Optional backend | targeted dependency-present and dependency-absent paths | backend/platform CI |
| Performance-sensitive loop | correctness tests and before/after benchmark | relevant benchmark environment |
| Dependency or workflow | targeted environment/config validation | affected CI workflow |

Unknown or cross-cutting paths receive the PR baseline rather than silently
selecting no tests.

## External numerical references

Every new estimator requires a permanent comparison with existing software.
Prefer live R `fixest` or another established package. Use
`against_r_core` when the reference is available on conda-forge and
`against_r_extended` for CRAN-only dependencies. Stored Stata or other
verified output is acceptable when its generator script and software version
are committed.

Any test file importing rpy2 must be listed in `_rpy2_test_files` in
`tests/conftest.py` so non-R environments skip it safely. HAC tests use
single-threaded BLAS to avoid oversubscription.

Compare named numerical quantities through the public API. Record deterministic
data or seeds, formulas, weights, vcov/SSC, package versions, and explicit
`rtol`/`atol` with a numerical justification. Add edge, brute-force,
closed-form, and simulation tests as needed, but never substitute them for the
external comparison required for a new estimator.

## Test design

Prefer a small number of heavily parametrized integration tests over many thin
wrapper tests. Extend an existing formula/vcov/weights/SSC matrix when the new
case fits it. Unit-test internal seams only when the public API cannot exercise
them cleanly.

For predictions, compare a small deterministic subset rather than an entire
vector. Cover singleton clusters, collinearity, tiny samples, invalid inputs,
and every supported weights/FE/IV/multiple-estimation path. Unsupported paths
must raise a specific informative error.
