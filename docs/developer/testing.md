# Testing and verification

Select evidence for the changed behavior and its risks. Use `pixi run` for
every Python, pytest, lint, docs, and R command; bare tools can miss the
compiled extension or optional dependencies. This file owns check selection,
numerical references, and reporting. Skills apply these decisions.

## Runtime tiers

Use targeted tests while editing. Once the implementation stabilizes, broaden
coverage once as required below. Reuse passing checks after changes that do
not invalidate them; preparing a PR is not a reason to rerun the same tests.

| Stage | Evidence |
|---|---|
| Editing | Targeted regression, relevant release-contract or live-R cases, changed-file lint, type check |
| Stabilized implementation | Affected subsystem and Python baseline when required |
| Before merge | Affected long suites and platform CI on the exact PR head SHA |
| Exhaustive/release | All available suites, explicitly installed CRAN references, platform CI, relevant benchmarks |

Do not repeatedly run canonical R suites or `test-all` while iterating. A
required long check may be deferred at implementation handoff to CI on the
exact head when targeted checks pass and another local run would add no
diagnostic value. Report the reason and destination. Never defer a failure or
a targeted check needed to resolve material uncertainty. Scheduled or running
CI is not a pass; the change is not merge-ready until required evidence is
green. An older-head CI run does not establish the final head's readiness.

`test-py` is the broad Python baseline, not proof of numerical correctness.
`test-all` collects what the current environment supports; CRAN-only tests
can skip unless their packages are installed. Neither replaces the affected
external comparisons selected below.

## Commands

Use `pixi task list` for the full task inventory.

```bash
# Targeted edit feedback, without a repository-wide coverage report
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov
pixi run -e py312-r pytest tests/test_vs_r_fast.py -q --no-cov -k feols
pixi run -e py312-r test-r-fixest-fast

# Invariance and broad Python regression
pixi run -e py312 test-release-contract
pixi run test-py

# Canonical external comparisons: choose affected suites
pixi run -e py312-r test-r-core
pixi run -e py312-r test-r-fixest
pixi run -e py312-r test-r-hac

# Exhaustive available coverage; install CRAN references explicitly
pixi run -e py312-r test-all
pixi run -e py312-r Rscript r_test_requirements.R
pixi run -e py312-r test-r-extended

# Changed Python files; ty checks the whole package
pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check --files <changed files>
pixi run ty

# Benchmarks: use shared storage when comparing different worktrees
pixi run -e py312 bench-estimation --benchmark-save=base
pixi run -e py312 bench-estimation --benchmark-compare

# Documentation, only when selected below
pixi run docs-build
pixi run docs-render
```

`test-r-core` aggregates the conda-forge R comparisons. CI partitions them
into `test-r-fixest` and `test-r-core-other`, excluding both
`test_vs_fixest.py` and `test_multiway_clustering_vs_fixest.py` from the latter.
It runs `test-r-python` for non-R tests in rpy2-importing files separately,
and the fast live-R suite as preflight in the canonical fixest job.

## Selection matrix

Apply all relevant rows, not every row. Unknown cross-cutting code paths need
the Python baseline rather than silently selecting no tests.

| Change | Required edit/handoff evidence | Required merge/long evidence |
|---|---|---|
| Contributor guidance, including `docs/developer/` | `git diff --check`; validate affected links and skills/templates/configuration | Affected workflow only; no docs build/render or numerical suites |
| Rendered prose under `docs/` | `git diff --check`; execute changed examples; render affected page when practical | Full `docs-render` only for site-wide configuration, navigation, templates, or cross-page changes |
| Public docstrings or reference configuration | `git diff --check`; execute changed examples; `docs-build` | Affected reference-page render when applicable |
| Python API or internals | Targeted tests through [core user-facing functions](#test-design), changed-file lint, `pixi run ty` | Python baseline |
| Invariant internal/backend refactor | Above, plus passed release contract and affected-path tests | Python baseline; applicable external suites for every touched estimator |
| Input preparation, indexing, or orchestration around an unchanged method | Independent regression and edge evidence; release contract with intentional differences declared by `reason`; lint and type check | Python baseline and existing applicable external suites |
| Estimation/inference equations or numerical algorithms | Targeted integration/edge tests; release contract with intentional differences declared | Applicable live external-reference suite |
| New estimator | Complete support matrix and permanent external comparison | Python baseline, external suite, full platform CI |
| HAC | Targeted HAC/meat tests | Single-threaded `test-r-hac` |
| Rust | Kernel/reference integration tests | Python baseline, platform CI, relevant benchmarks |
| Optional backend | Dependency-present and dependency-absent paths | Backend/platform CI |
| Demeaning, IRLS, solvers, vcov, model-matrix construction, Rust kernels, or other performance-sensitive loops | Correctness tests and before/after benchmark | Relevant benchmark environment |
| Dependency or workflow | Targeted environment/configuration validation | Affected CI workflow |

`docs-build` regenerates API-reference inputs; it is not a Markdown validator.
Do not run it for contributor guidance or workflow metadata. Local required
checks must pass before handoff, subject only to the long-check deferral above.

## Estimation benchmarks

Run `bench-estimation` on the merge-base and branch for the performance paths
in the matrix. It times OLS, Poisson, logit, and IV on simple and difficult
`base_dgp` panels with the within LSMR demeaner and additive/diagonal
preconditioners. The opt-in, single-threaded suite takes about two minutes and
does not run in CI. It measures speed, not numerical agreement.

Save the baseline with `--benchmark-save=base`, then compare on the branch
with `--benchmark-compare`. Worktrees need the same `--benchmark-storage`:
the default `.benchmarks/` is local to each working directory. Use an otherwise
idle machine. Rerun both sides of any case more than 15% slower before
reporting a regression; millisecond cases can vary by 20% under load. Report
median ratios for affected case groups and explain confirmed slowdowns.

## Verification reporting

Account for applicable checks as **passed**, **failed**, **deferred**, or
**not run**. Group successes in a concise sentence naming the suites or test
selections. Identify evidence from CI on the exact head; local and CI runs of
the same check are not independent evidence. Do not reproduce a second
checklist or a command-by-command success log in the handoff or PR.

For failures, deferrals, and missing checks, give the command, reason, and
unresolved risk. Deferrals also need their destination and head SHA. Include
elapsed time when it explains a deferral or supports a runtime claim. Always
give the release-contract passed count or skip reason: skips are not passes.
Distinguish review-ready work with outstanding merge evidence from a
merge-ready change.

## Release contract

`tests/test_release_contract.py` compares public results with a pinned pyfixest
release using shared deterministic cases. It detects drift, not pre-existing
bugs in that release. `test-release-contract` records a missing/stale baseline
before testing; `test-py` uses a valid existing baseline or skips those cases.
CI does not record it, so canonical R suites remain its merge evidence.

For an invariant refactor, run the contract early and after result-affecting
edits. Unexplained drift is a regression. Fix it or explicitly reclassify the
change as numerical. An intentional change needs a `reason` in the contract,
an independent regression test under the next section, and a changelog entry.
Widen an individual `baseline.check(...)` or use `baseline.skip(...)` only for
a documented behavior change since the pin, never simply to pass a failure.
Default comparisons are near machine precision.

## Choosing a numerical reference

Independence means bypassing the faulty or modified path, not necessarily
using a different package. For input preparation, formula rebuilding,
algorithm selection, indexing, or orchestration around an unchanged
estimator, prefer an explicit calculation or direct public-API refit when:

- the estimator and statistical definition stay unchanged;
- the reference explicitly constructs expected inputs and bypasses the path;
- the test would fail on the buggy implementation.

This can suffice even when fixing the bug changes numerical results. For
example, test `ritest()` rebuilding by permuting raw treatment data, refitting
with `feols()`, and comparing treatment coefficients. Comparing two `ritest()`
modes that share the rebuilding code is insufficient.

New estimators need permanent external comparisons. Changes to equations or
numerical algorithms need external evidence wherever overlapping software
exists, as does a comparison that could share the numerical error under test.
State briefly what makes the reference independent. An adequate internal
regression needs no duplicate external test, but existing compatibility
coverage and applicable external suites remain required. Keep additional
external coverage when it tests a distinct uncertainty, such as sample
selection or covariance behavior absent from the internal reference.

## External numerical references

When external evidence is required, prefer live R `fixest`, another
established R package, or an established Python package in a maintained
environment. Next prefer CRAN-only R references, then stored Stata/other
established output with its generating script and exact version. Use stored
results only when the live implementation is unavailable/unreliable in the
environment or measured per-case runtime makes regular runs impractical.
First try affected live cases instead of replacing a reference merely to
avoid a full suite during editing. Simulations, shapes, and internal
reimplementations do not substitute for required external evidence.

Use `against_r_core` for conda-forge references and `against_r_extended` for
CRAN-only dependencies. List every rpy2-importing file in `_rpy2_test_files`
in `tests/conftest.py`, with the strict marker matching its dependencies.
HAC uses single-threaded BLAS.

Both sides need the same deterministic rows and specification. Record seeds
or data, formulas, weights, vcov/SSC, package versions, and justified explicit
tolerances. Compare named quantities through public APIs: coefficients, vcov,
standard errors, degrees of freedom, observations, dropped terms, convergence,
and deterministic prediction subsets as relevant. Supplement with edge,
closed-form, brute-force, or simulation checks where useful.

The fast suite in `tests/test_vs_r_fast.py` is edit feedback, not complete
merge evidence or a separate permanent reference framework. Extend canonical
coverage for new behavior. Quantreg has no absorbed FE support; its inference
and tolerances follow R `quantreg`, not the fixest IID/heteroskedastic/CRV1 matrix.

### Tolerance contract

Name the failed quantity in numerical assertions' `err_msg`. Align coefficient
names and covariance rows/columns; compare discrete structure exactly.
Choose separate tolerances for coefficients, vcov/inference, residuals, and
predictions, with coefficients normally strictest. Explain looser tolerances
next to the assertion through a concrete numerical cause, such as stopping
error, FE recovery, or clustered accumulation order. “Implementations differ”
is not a justification. Do not copy the loosest tolerance from another test.

Use `tests/test_vs_fixest.py` for current OLS/Poisson/GLM comparison standards
and `tests/test_quantreg.py` for quantreg's solver-specific standards.

## Test design

Prefer tests through the core user-facing functions, such as `feols`, `feglm`,
`fepois`, `quantreg`, `etable`, `coefplot`, and `summary`. This is what “tests
through the public API” means in this guide. Fit models or produce tables,
plots, and summaries through the relevant entry point, then assert the
resulting behavior.

Use an internal seam when it isolates a meaningful contract these entry points
cannot exercise cleanly, not merely because a helper was added. Before adding
coverage, check whether an existing test already distinguishes the regression.
Otherwise extend a coherent parametrized test with the smallest discriminating case. Use a new
function or file when its setup, assertion, dependency, or fixture lifecycle
warrants one. Do not force a distinct regression into a huge Cartesian matrix.

New or changed errors/warnings need triggering tests in `tests/test_errors.py`
or the nearest subsystem suite. Assert category and stable message text with
`pytest.raises(..., match=...)` or `pytest.warns(..., match=...)`.

Reuse seeded fixtures, external adapters, and assertion helpers. Explain a
new test file or unusually large test diff by the behavior it distinguishes
and its runtime impact, not a line-count target. Check relevant singleton,
collinearity, tiny-sample, invalid-input, weights/FE/IV/multiple-estimation
paths; unsupported paths must fail informatively. For predictions/residuals,
prefer a small deterministic subset with quantity-specific tolerances.

## Release-baseline maintenance

The recorder uses the pinned release wheel in `tests/snapshots/release/`.
Recordings are gitignored and local to a worktree/platform. The test file,
baseline module, shared cases, release lockfile, and platform fingerprint
invalidate stale recordings. To record without testing the checkout:

```bash
pixi run --locked --manifest-path tests/snapshots/release/pixi.toml record
```

Use pixi 0.71.0 or newer for the v7 lockfile. Do not add `--clean-env` (unsupported
on Windows); `scripts/record_release_baseline.py` guards the release import.
Roll the pin only deliberately, normally just after a stable release, through
the task below. Reassess declared differences; the suite warns about newer tags.

```bash
pixi run roll-release-baseline          # newest release tag in this checkout
pixi run roll-release-baseline 0.61.0   # explicit release
```
