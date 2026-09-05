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
| Edit feedback | Exercise the changed seam repeatedly | Seconds to a few minutes | targeted pytest, release contract, targeted or fast live-R checks, changed-file Ruff/mypy |
| Stabilized implementation | Broaden local confidence once | Minutes | selected subsystem checks, `pixi run test-py` when required |
| Merge evidence | Validate the exact PR head in affected environments | May take tens of minutes | canonical R, HAC, no-JIT, docs, plots, Rust, platform CI |
| Exhaustive or release | Exercise everything available | Potentially substantially longer | `test-all`, CRAN-only dependencies, platform CI, benchmarks |

During a related batch of initial edits or review feedback, run the focused
behavior and caller tests plus lightweight checks that can expose the affected
seam. Once the batch stabilizes, satisfy its selected requirements with evidence
that still applies and run missing or invalidated checks. A new edit, commit,
review comment, agent handoff, push, or PR opening is not by itself a reason to
repeat a check. Rerun evidence when later work invalidates it, a material risk
remains unresolved, or an explicit acceptance requirement calls for a newer
run. Do not repeatedly launch `test-r-fixest`, `test-r-core`, or `test-all`
while iterating.

Coherent layers that work against their immediate parent need their applicable
focused checks, not the full acceptance suite after every edit. For a stack or
other multi-batch change, coordinate one owner for each broad check and normally
run it once on the stabilized cumulative head against the trunk. Run broad
checks per layer only when the layers must be independently releasable or an
acceptance requirement says so. A required long check may be deferred to
exact-head CI at implementation handoff when the report names the check and
destination.
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
| Public docstrings or API-reference configuration | `git diff --check`; execute changed examples; `docs-build` once stabilized | affected reference-page render when applicable |
| Content under `docs/` | `git diff --check`; execute changed examples; render the affected page when practical | `docs-render` only for site-wide configuration, navigation, templates, or cross-page changes |
| Repository guidance or workflow metadata outside `docs/` | `git diff --check`; targeted skill, template, or configuration validation | affected CI workflow only; no docs build by default |
| Python API or internals | targeted public tests; changed-file lint/type checks | Python baseline |
| Internal or backend refactor with unchanged results | release contract green (passed, not skipped); targeted tests; changed-file lint/type checks | Python baseline; applicable external suite for every estimator the refactor touches |
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
`AGENTS.md`, `.agents/`, `.github/` templates, or contributor workflow metadata.
Do not run it for plain prose or changelog changes either. Where new evidence is
needed, run it once after a change to public docstrings, API-reference
configuration, or documentation imports has stabilized. For changed examples or
prose under `docs/`, execute or render only the affected evidence where useful.
Reserve the full `docs-render` task for changes that can affect the site broadly,
and rebuild affected evidence only when a later edit invalidates it.

## Release contract

`tests/test_release_contract.py` mirrors the structure of
`tests/test_vs_fixest.py` — the same data fixtures, the same parametrization
over the shared formula tuples in `tests/_feols_test_cases.py`, one test per
estimator — but replaces the R reference with results recorded from a pinned
pyfixest release. It is a fast regression alarm, not an external correctness
oracle: a bug already present in the pinned release is recorded, not caught.

The baseline is recorded by running that same test file under the release
wheel, in the locked workspace in `tests/snapshots/release/`, so the two sides
cannot describe different case matrices. `test-release-contract` records it on
first use and reuses it afterwards; the recording is platform-local and
gitignored, so every operating system and architecture compares against its own
floating-point output. `test-py` picks the suite up once a baseline exists and
skips it otherwise, so it never forces a recording. CI deliberately does not
record a baseline, so the suite is local-only: it skips on every runner, and the
canonical R suites remain the exact-head merge evidence. A fingerprint over the test file, the baseline module,
the shared case lists, the release lockfile, and the platform invalidates it
automatically. To record it without running the checkout's tests:

```bash
pixi run --locked --manifest-path tests/snapshots/release/pixi.toml record
```

The nested workspace's lockfile is format v7, like the checkout's own, so
recording needs pixi 0.71.0 or newer, the same minimum as the rest of the
repository. It deliberately avoids `--clean-env`, which pixi does not support
on Windows; `scripts/record_release_baseline.py` guards the release import
itself.

The pinned release lives in one place, the `pyfixest` entry of
`tests/snapshots/release/pixi.toml`. Roll it just after tagging a release --
that is what brings back the comparisons the documented differences currently
skip, and the skip list is shortest right then:

```bash
pixi run roll-release-baseline          # newest release tag in this checkout
pixi run roll-release-baseline 0.61.0   # a specific release
```

The suite warns when a newer release tag exists than the pinned version.

Comparisons use a near-machine-precision default. Widen a single
`baseline.check(...)` call, or `baseline.skip(...)` a quantity, only for a
behaviour change that post-dates the pinned release, and give the call an
explicit `reason`; unexplained drift is a regression for human review, not a
tolerance to raise. Change the pin in `tests/snapshots/release/pixi.toml` only
through `roll-release-baseline`, for a deliberate roll to a stable release.

The recording lives under the checkout root, so each worktree records its own
baseline. When the suite fails after a change you intended as a pure refactor,
the change is no longer a refactor: either fix the regression, or reclassify it
as a numerics change. Declare the difference in
`tests/test_release_contract.py` with a `reason`, add the external comparison
the "Estimation or inference numerics" row requires, and record it in the
changelog.

For an invariant internal or backend refactor, satisfy the release-contract
requirement when the related change batch stabilizes using still-applicable
evidence. Where a new run is needed, run the full contract normally once on the
cumulative head. Run it earlier when shared numerical risk is not covered by
focused tests, or when the contract is needed to investigate the change or
establish a baseline. It is not an automatic gate for every iteration.

## Evidence records

Keep the verification record compact. Record the revision, environment, scope,
command, and result for material checks, and name one owner for each coordinated
broad check. Evidence from an earlier revision may remain relevant when the
later diff cannot affect it, but do not describe it as exact-head evidence.
State what changed since the run and rerun only the evidence that change
invalidated. Changes to dependencies, environments, or test configuration can
invalidate a result even when the tested source is unchanged. Final merge
requirements and exact-head CI gates remain unchanged.

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
