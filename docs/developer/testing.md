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
| Full available suite | All tests supported by the current environment | Potentially substantially longer | `test-all`, platform CI, benchmarks |

Run edit checks repeatedly. Run required domain checks once the design is
stable. A verification report records actual elapsed time and reports every
applicable check as passed, failed, deferred, or not run. Deferred is never
equivalent to passed.

`test-all` collects every test supported by the current Pixi environment. It
does not install the CRAN-only packages from `r_test_requirements.R`, so those
tests can skip. Exhaustive release evidence requires installing those packages,
running `test-r-extended`, and combining the available suite with the relevant
platform CI and benchmarks.

## Commands

```bash
# Targeted, without the repository-wide coverage report
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov

# Python baseline
pixi run test-py

# External references
pixi run test-r-core
pixi run test-r-fixest
pixi run -e py312-r test-r-fixest-fast
pixi run test-r-hac
pixi run test-r-extended

# All tests available in the current environment
pixi run test-all

# CRAN-only reference tests require an explicit installation first
pixi run -e py312-r Rscript r_test_requirements.R
pixi run test-r-extended

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
| Documentation only | `git diff --check`; execute changed examples; `docs-build` | affected-page render or `docs-render` when applicable |
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

`pixi run -e py312-r test-r-fixest-fast` runs representative rpy2 cases
directly against R `fixest` for `feols`, `fepois`, and `feglm`, and R
`quantreg` for `quantreg`. Use `-k feols`, `-k fepois`, `-k feglm`, or
`-k quantreg` to select one entry point while editing. This is a focused edit
check, not a second permanent reference framework; extend the canonical suites
when a change needs new coverage.

### Error and tolerance contract

For an `assert_allclose(actual, reference, rtol, atol)` comparison, the tested
elementwise error is

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Thus `atol` governs values near zero and `rtol` governs differences at the
scale of the reference value. Every numerical assertion must identify the
quantity that failed in `err_msg`. Align named coefficients and covariance
rows/columns before comparing them; compare observation counts, degrees of
freedom, dropped-term sets, and other discrete structure exactly.

Do not prescribe one tolerance for every estimator or copy the loosest
tolerance in a test. Use separate, numerically justified tolerances for
coefficients, vcov/standard errors and derived inference, residuals, and
predictions. Coefficients normally receive the strictest tolerance. Iterative
algorithms, fixed-effect recovery, and cluster inference may require looser
tolerances, which must be explained next to the assertion.

Treat `tests/test_vs_fixest.py` as the source of truth for the current
`feols`, `fepois`, and `feglm` comparison standards, including the formulas,
parameters, quantities, and absolute-error tolerances being tested. Follow
`tests/test_quantreg.py` for `quantreg`, whose solver-specific relative and
absolute tolerances are different. The compact `tests/test_vs_r_fast.py` suite
uses representative combinations from those contracts rather than duplicating
their complete matrices.

## Test design

Prefer a small number of heavily parametrized integration tests over many thin
wrapper tests. Extend an existing formula/vcov/weights/SSC matrix when the new
case fits it. Unit-test internal seams only when the public API cannot exercise
them cleanly.

Every behavioral change needs regression evidence, but it does not necessarily
need a new test. Control suite growth in this order:

1. If an existing test already exercises the changed behavior, identify it and
   do not duplicate it.
2. If an existing parametrized matrix can represent the regression, add the
   smallest case that would catch it.
3. Add a test function only when the setup or assertion contract is genuinely
   different.
4. Add a test file only for a distinct subsystem, dependency marker, or fixture
   lifecycle.

Reuse seeded fixtures, external-reference adapters, and assertion helpers. A
new test file or unusually large test diff must explain why an existing matrix
cannot cover the behavior coherently and report the runtime impact. Do not add
duplicate coverage merely to give one edge case its own test function. Avoid
hard line-count limits: review whether the test remains legible and whether its
maintenance and runtime cost are proportional to the regression it prevents.

For predictions and residuals, compare a small deterministic subset rather
than an entire vector and give each quantity its own tolerance. Cover singleton
clusters, collinearity, tiny samples, invalid inputs, and every supported
weights/FE/IV/multiple-estimation path. Unsupported paths must raise a specific
informative error.
