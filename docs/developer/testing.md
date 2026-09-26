# Testing and verification

Pyfixest testing is risk-based: use fast checks while editing, broaden the
evidence once the implementation stabilizes, and require the affected
long-running suites before merge. Always use `pixi run`; bare Python or pytest
may not have the compiled extension or the right optional dependencies.

## Contents

- [Runtime tiers](#runtime-tiers): edit, stabilized, merge, and exhaustive
  stages; what exact-head CI means; deferral semantics
- [Commands](#commands): the `pixi run` tasks for each stage
- [Selection matrix](#selection-matrix): which checks each kind of change
  requires
- [Verification reporting](#verification-reporting): concise evidence and status
- [Release contract](#release-contract): regression checks and declared differences
- [External numerical references](#external-numerical-references): reference
  preference order, rpy2 rules, and the [tolerance contract](#tolerance-contract)
- [Test design](#test-design): controlling suite growth, error-path tests,
  fixtures, and edge coverage
- [Release-baseline maintenance](#release-baseline-maintenance): recording and
  rolling the pin

## Runtime tiers

Runtime labels are qualitative because machine, compiler cache, and test
selection affect wall time.

| Stage | Purpose | Typical scale | Examples |
|---|---|---|---|
| Edit feedback | Exercise the changed seam repeatedly | Seconds to a few minutes | targeted pytest, release contract, targeted or fast live-R checks, changed-file Ruff, `pixi run ty` |
| Stabilized implementation | Broaden local confidence once | Minutes | selected subsystem checks, `pixi run test-py` when required |
| Merge evidence | Validate the exact PR head in affected environments | May take tens of minutes | canonical R, HAC, no-JIT, docs, plots, Rust, platform CI |
| Exhaustive or release | Exercise everything available | Potentially substantially longer | `test-all`, CRAN-only dependencies, platform CI, benchmarks |

Merge evidence must run against the exact PR head SHA. This document and the
skills call a CI run on that SHA *exact-head CI*; a run on an older head is
stale and does not count.

Run edit checks repeatedly, but do not repeatedly launch `test-r-fixest`,
`test-r-core`, or `test-all` while iterating. Use a targeted test or
`test-r-fixest-fast` instead. Once the design is stable, run the selected
baseline once. A required long check may be deferred to exact-head CI at
implementation handoff when a local run would add no diagnostic value and
targeted checks have passed, and the report follows "Verification reporting".
Never defer a failing check or a targeted check needed to resolve a material
uncertainty. The change is not merge-ready until all required merge evidence
is green.

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

# Release contract: regression alarm against the pinned pyfixest release
pixi run -e py312 test-release-contract

# Targeted live-R edit feedback
pixi run -e py312-r pytest -q tests/test_vs_r_fast.py --no-cov -k feols
pixi run -e py312-r test-r-fixest-fast

# Python baseline
pixi run test-py

# Canonical external-reference evidence
pixi run -e py312-r test-r-core
pixi run -e py312-r test-r-fixest
pixi run -e py312-r test-r-hac

# All tests available in the current environment
pixi run -e py312-r test-all

# CRAN-only reference tests require an explicit installation first
pixi run -e py312-r Rscript r_test_requirements.R
pixi run -e py312-r test-r-extended

# Changed-file quality checks
pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check --files <changed files>

# Whole-package type check (ty has no changed-file mode)
pixi run ty

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
| Rendered content under `docs/` (excluding `docs/developer/`) | `git diff --check`; execute changed examples; render the affected page when practical | `docs-render` only for site-wide configuration, navigation, templates, or cross-page changes |
| Repository guidance (including `docs/developer/`) or workflow metadata | `git diff --check`; validate changed links and applicable skills, templates, or configuration | affected CI workflow only; no docs build or render |
| Python API or internals | targeted public tests; changed-file lint; `pixi run ty` | Python baseline |
| Internal or backend refactor with unchanged results | release contract green (passed, not skipped); targeted tests; changed-file lint; `pixi run ty` | Python baseline; applicable external suite for every estimator the refactor touches |
| Estimation or inference numerics | targeted integration and edge tests; release contract with every intended difference declared by `reason` | applicable live external-reference suite |
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
`AGENTS.md`, `.agents/`, `docs/developer/`, `.github/` templates, or contributor
workflow metadata. For rendered prose under `docs/`, prefer an affected-page
render. Reserve the full `docs-render` task for changes that can affect the site
broadly.

## Verification reporting

Account for every applicable check as **passed**, **failed**, **deferred**, or
**not run**. Group successful checks in a concise sentence with recognizable
suite names or test selections; link detailed output when available. Identify
which results came from CI on the exact head. Do not count duplicate local and
CI runs as independent evidence.

Give exact commands and details for failures, deferrals, and checks not run:
reason, unresolved risk, and, for deferrals, destination and head SHA. Include
elapsed time when it explains a deferral or supports a runtime claim. For the
release contract, always give the passed case count or skip reason; skipped
comparisons are not passes. Deferred or merely scheduled CI is not a pass.

Use this format in handoffs and PR bodies without a second checklist or a
command-by-command success log. Required local checks must pass and be reported
before implementation handoff. All required merge evidence must pass on the
exact PR head before claiming merge readiness.

## Release contract

`tests/test_release_contract.py` compares public estimator results with a
pinned pyfixest release, using the same fixtures and shared formula matrix as
its recorder. This is an invariance check, not an external correctness oracle:
existing release bugs are recorded, not caught.

`pixi run -e py312 test-release-contract` records a missing or stale baseline
before testing. `test-py` runs these comparisons only when a valid baseline
exists and skips the release-contract cases otherwise; it never records one.
CI does not record a baseline, so these cases skip there and the canonical R
suites remain the merge evidence. Only passed comparisons establish invariance.

For an invariant refactor, run the contract early and after edits that can
affect results. Reuse evidence after unrelated prose or metadata edits.
Unexplained drift is a regression: fix it, or explicitly reclassify the change
as numerics. An intentional change needs a `reason` in
`tests/test_release_contract.py`, the applicable external comparison, and a
changelog entry. Widen an individual `baseline.check(...)` or use
`baseline.skip(...)` only for a documented behavior change since the pinned
release, never simply to make a failure pass. The default comparison is near
machine precision.

For baseline setup, invalidation, or release rolls, read
[Release-baseline maintenance](#release-baseline-maintenance).

## External numerical references

Every new estimator requires a permanent comparison with existing software;
numerical changes to existing estimators require one wherever overlapping
software exists. Simulations, shape checks, and internal reimplementations do
not substitute for external evidence. Choose the reference in this order:

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

## Release-baseline maintenance

The recorder runs the same test file under the pinned release wheel in the
locked workspace `tests/snapshots/release/`. Recordings are gitignored and local
to each worktree and platform. A fingerprint of the test file, baseline module,
shared case lists, release lockfile, and platform invalidates stale recordings.
To record without running the checkout's tests:

```bash
pixi run --locked --manifest-path tests/snapshots/release/pixi.toml record
```

Recording requires pixi 0.71.0 or newer for the v7 lockfile. Do not add
`--clean-env`, which pixi does not support on Windows;
`scripts/record_release_baseline.py` guards the release import.

The pin lives in the `pyfixest` entry of `tests/snapshots/release/pixi.toml`.
Change it only through `roll-release-baseline`, for a deliberate roll to a
stable release. Roll just after tagging a release, then reassess the declared
differences; the suite warns when a newer release tag exists than the pin.

```bash
pixi run roll-release-baseline          # newest release tag in this checkout
pixi run roll-release-baseline 0.61.0   # a specific release
```
