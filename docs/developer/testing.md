# Testing and verification

Pyfixest testing is risk-based: use fast checks while editing, broaden the
evidence once the implementation stabilizes, and require the affected
long-running suites before merge. Always use `pixi run`; bare Python or pytest
may not have the compiled extension or the right optional dependencies.

## Runtime tiers

Runtime labels are qualitative because machine, compiler cache, and test
selection affect wall time.

| Stage | Purpose | Typical scale | Examples |
|---|---|---|---|
| Edit feedback | Exercise the changed seam repeatedly | Seconds to a few minutes | targeted pytest, targeted or fast live-R checks, changed-file Ruff/mypy |
| Stabilized implementation | Broaden local confidence once | Minutes | selected subsystem checks, `pixi run test-py` when required |
| Merge evidence | Validate the exact PR head in affected environments | May take tens of minutes | canonical R, HAC, no-JIT, docs, plots, Rust, platform CI |
| Exhaustive or release | Exercise everything available | Potentially substantially longer | `test-all`, CRAN-only dependencies, platform CI, benchmarks |

Run edit checks repeatedly, but do not repeatedly launch `test-r-fixest`,
`test-r-core`, or `test-all` while iterating. Use a targeted test or
`test-r-fixest-fast` instead. Once the design is stable, run the selected
baseline once. A required long check may be deferred to exact-head CI at
implementation handoff when the report names the check and destination.
Deferred is never equivalent to passed, and the change is not merge-ready
until all required merge evidence is green.

`test-py` is the broad Python-only regression baseline. It does not compare
results with R or another external implementation, so it cannot establish
numerical parity for estimation or inference changes. Run the applicable
external-reference suite as separate, stronger evidence.

`test-all` collects every test supported by the current Pixi environment. It
does not install the CRAN-only packages from `r_test_requirements.R`, so those
tests can skip. Exhaustive release evidence requires installing those packages,
running `test-r-extended`, and combining the available suite with the relevant
platform CI and benchmarks.

## Commands

```bash
# Targeted, without the repository-wide coverage report
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov

# Targeted live-R edit feedback
pixi run -e py312-r pytest -q tests/test_vs_r_fast.py --no-cov -k feols
pixi run -e py312-r test-r-fixest-fast

# Python baseline
pixi run test-py

# Canonical external-reference evidence
pixi run -e py312-r test-r-core
pixi run -e py312-r test-r-fixest
pixi run -e py312-r test-r-hac
pixi run -e py312-r test-r-extended

# All tests available in the current environment
pixi run -e py312-r test-all

# CRAN-only reference tests require an explicit installation first
pixi run -e py312-r Rscript r_test_requirements.R
pixi run -e py312-r test-r-extended

# Changed-file quality checks
pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check --files <changed files>
pixi run -e lint prek run mypy --files <changed files>

# Documentation (costly; run only when the selection matrix calls for it)
pixi run docs-build
pixi run docs-render
```

`test-r-core` remains the convenient local aggregate for all canonical
conda-forge R comparisons. CI partitions that population into
`test-r-fixest` and the internal `test-r-core-other` shard so
`tests/test_vs_fixest.py` is not executed twice. The fast suite runs once as a
preflight in the canonical `fixest` job.

## Selection matrix

This matrix is authoritative for which checks a change requires.

| Change | Required edit or handoff evidence | Merge or long evidence |
|---|---|---|
| Public docstrings or API-reference configuration | `git diff --check`; execute changed examples; `docs-build` | affected reference-page render when applicable |
| Content under `docs/` | `git diff --check`; execute changed examples; render the affected page when practical | `docs-render` only for site-wide configuration, navigation, templates, or cross-page changes |
| Repository guidance or workflow metadata outside `docs/` | `git diff --check`; targeted skill, template, or configuration validation | affected CI workflow only; no docs build by default |
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

Documentation builds are opt-in. `docs-build` regenerates API-reference inputs;
it is not a general Markdown validator. Do not run it for changes limited to
`AGENTS.md`, `.agents/`, `.github/` templates, or contributor workflow metadata.
For prose under `docs/`, prefer an affected-page render. Reserve the full
`docs-render` task for changes that can affect the site broadly.

## External numerical references

Every new estimator requires a permanent comparison with existing software.
Choose the reference in this order:

1. live R `fixest`, another established R package, or a well-established Python
   package available in a maintained environment;
2. a CRAN-only R package in the extended environment;
3. stored output from Stata or other established software, with the generating
   script and exact version committed;
4. another established external package with its exact version recorded.

Use `against_r_core` when the reference is available on conda-forge and
`against_r_extended` for CRAN-only dependencies. Do not replace an available
live reference merely to avoid running the complete canonical suite while
editing; first select the affected live cases or use the fast matrix. Reach for
a stored result only when the external implementation is unavailable or
unreliable in the test environment, or when measured per-case runtime makes
regular execution impractical.

Any test file importing rpy2 must be listed in `_rpy2_test_files` in
`tests/conftest.py` so non-R environments skip it safely, and must use the
strict R marker matching dependency availability. HAC tests use single-threaded
BLAS to avoid oversubscription.

Use the same deterministic rows and model specification on both sides. Compare
named numerical quantities through the public API so ordering differences cannot
hide discrepancies. Record deterministic data or seeds, formulas, weights,
vcov/SSC, package versions, and explicit `rtol`/`atol` with a numerical
justification. Compare the quantities the method promises: coefficients, vcov,
standard errors, degrees of freedom, observations, dropped terms, convergence,
or deterministic prediction subsets as applicable. Add edge, brute-force,
closed-form, and simulation tests as needed, but never substitute them for the
external comparison required for a new estimator.

`pixi run -e py312-r test-r-fixest-fast` runs representative rpy2 cases directly
against R `fixest` for `feols`, `fepois`, and `feglm`, and R `quantreg` for
`quantreg`. See `tests/test_vs_r_fast.py` for the exact cases it covers. This is
edit feedback, not a second permanent reference framework or complete merge
evidence; extend the canonical suites when a change needs new coverage.
`quantreg` does not support fixed effects, and its inference contract follows R
`quantreg` rather than the `fixest` IID/heteroskedastic/CRV1 matrix.

### Tolerance contract

Every numerical assertion must identify the quantity that failed in `err_msg`.
Align named coefficients and covariance rows/columns before comparing them;
compare observation counts, degrees of freedom, dropped-term sets, and other
discrete structure exactly.

Do not prescribe one tolerance for every estimator or copy the loosest
tolerance in a test. Use separate, numerically justified tolerances for
coefficients, vcov/standard errors and derived inference, residuals, and
predictions. Coefficients normally receive the strictest tolerance. Iterative
algorithms, fixed-effect recovery, and cluster inference may require looser
tolerances. A looser tolerance must be explained next to the assertion with a
specific numerical reason, such as solver stopping error, fixed-effect recovery
error, or floating-point accumulation order in clustered reductions. Merely
noting that two implementations differ is not a justification.

Treat `tests/test_vs_fixest.py` as the source of truth for the current
`feols`, `fepois`, and `feglm` comparison standards, including the formulas,
parameters, quantities, and absolute-error tolerances being tested. Follow
`tests/test_quantreg.py` for `quantreg`, whose solver-specific relative and
absolute tolerances are different.

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

Every new or changed error or warning path needs a test that triggers it.
Extend `tests/test_errors.py` or the nearest subsystem suite, and assert the
exception or warning category plus stable message text with
`pytest.raises(..., match=...)` or `pytest.warns(..., match=...)`.

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
