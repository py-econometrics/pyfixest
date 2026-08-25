# pyfixest — guide for coding agents

pyfixest ports R's `fixest` to Python: high-dimensional fixed-effects estimation
(OLS/WLS, IV, Poisson, GLM, quantile regression), fixest formula syntax,
post-estimation tools, and Rust kernels for hot loops.

Two rules beat everything else:

1. **Mirror `fixest`** in user-facing behavior, naming, and defaults unless an
   intentional difference is documented and tested.
2. **Mirror the nearest existing implementation.** Find the in-repo precedent
   and copy its structure before writing new code.

`CLAUDE.md` is a committed thin redirect to this file; do not duplicate these
rules in tool-specific configuration.

## Required workflow skills

Load the repository-local skill when its trigger applies:

| Skill | Trigger |
|---|---|
| [`implementation-strategy`](.agents/skills/implementation-strategy/SKILL.md) | Before estimator, public API, inference, formula, or shared-core work |
| [`numerical-validation`](.agents/skills/numerical-validation/SKILL.md) | Any estimator or numerical-behavior change |
| [`change-verification`](.agents/skills/change-verification/SKILL.md) | Before handing off code, tests, docs, CI, or metadata changes |
| [`history-curation`](.agents/skills/history-curation/SKILL.md) | Before the first PR submission when agent-owned commits may need rewriting |
| [`pyfixest-pr-review`](.agents/skills/pyfixest-pr-review/SKILL.md) | PR review and final self-review |
| [`pr-draft-summary`](.agents/skills/pr-draft-summary/SKILL.md) | Preparing a single or stacked draft PR |

The authoritative architecture, test tiers, and compatibility ledger live in
[`docs/developer/`](docs/developer/). Skills route work to those sources; do
not copy their policy into ad hoc prompts.

## Architecture strategy

Keep the shared estimation core narrow and stable. Formula parsing, model-matrix
construction, demeaning, generic fit and inference primitives, result
interfaces, and backend kernels are core. New estimators start as standalone
add-on functions in their own API or domain modules and compose those stable
primitives. Do not add estimator-specific switches to generic runners or grow
model classes with numerical logic.

A new primitive belongs in shared core only when it has a real shared consumer,
a generic contract, and maintainer design approval. A compatible estimator may
return an existing result type; otherwise give it a dedicated result class.
Post-estimation methods validate and delegate to standalone numerical functions.

## Repo map

| Path | Contents |
|---|---|
| `pyfixest/estimation/api/` | Public estimation entry points, one module per function |
| `pyfixest/estimation/models/` | Model/result classes; modules end in `_` |
| `pyfixest/estimation/internals/` | Shared fit, solver, vcov, collinearity, and separation primitives |
| `pyfixest/estimation/post_estimation/` | Standalone post-estimation logic |
| `pyfixest/estimation/formula/` | Formula parsing and model-matrix construction |
| `pyfixest/estimation/config.py`, `plan_.py`, `runner.py` | Configuration, formula planning, and estimation orchestration |
| `pyfixest/demeaners.py` | Public demeaner configurations |
| `pyfixest/core/`, `src/` | Python wrappers/type stubs and PyO3 Rust kernels |
| `pyfixest/did/`, `report/`, `utils/` | DiD estimators, reporting, utilities, and DGPs |
| `tests/` | Pytest suite, reference scripts, and stored reference outputs |
| `docs/` | Quarto user documentation and developer policy |

Estimation flow is `feols()` → `EstimationConfig` → `parse_formula` →
`runner.run_estimation` / `fit_one` → `prepare_model_matrix` → `get_fit` →
`vcov` → `get_inference`. `get_fit` performs `demean` → `to_array` →
`drop_multicol_vars` → `wls_transform` before solving. `FixestMulti` is a
container; post-estimation happens through fitted results.

## Where new code goes

- **Estimator:** a standalone module under `estimation/api/` (or the relevant
  domain package), with its own tests, result contract, exports, and quartodoc
  registration. Keep its special cases out of the generic fit pipeline.
- **Post-estimation feature:** numerical logic in `post_estimation/`; a thin
  method on `Feols` and siblings where applicable. Template:
  `post_estimation/ritest.py` and `Feols.ritest`.
- **Vcov type:** literal in `internals/literals.py`, validation in the model,
  small dispatch method, math in `internals/vcov_utils.py` or Rust, and wiring
  through `FixestMulti`/quantreg where supported. Template: NW/DK HAC.
- **Estimation-time option:** typed literal, early API validation,
  `EstimationConfig`, and `plan_._build_model_kwargs`.
- **Rust kernel:** `src/<topic>.rs`, registration in `src/lib.rs`, typed stub
  in `core/_core_impl.pyi`, and a clean wrapper in `core/`. Keep a NumPy
  reference implementation when feasible. Template: `src/nw.rs` →
  `core/nw.py`.

Reuse formula handling, `capture_context`, `_narwhals_to_pandas`, cluster
preparation, `run_crv_loop`, and `_create_rng`; do not rederive them.

## Numerical and code conventions

- Write for an econometrics practitioner. Use paper notation (`scores`, `meat`,
  `bread`, `u_hat`, `clustid`) and cite the paper in the implementing
  function.
- Methods orchestrate; numbers happen in standalone functions operating on
  arrays. Return small typed result dataclasses rather than tuples or dicts.
- Keep functions short and single-purpose. Solver iteration loops are the main
  exception when splitting would obscure the algorithm or hurt compilation.
- Put measured, non-vectorizable hot loops in Rust. Keep everything else clear,
  ordinary NumPy rather than speculative micro-optimization.
- Use `from __future__ import annotations`, PEP 604 unions, `Literal` aliases,
  keyword arguments for internal calls, and `NDArray[np.float64]` in stubs.
- Validate at the API boundary. Bad option values raise `ValueError` with the
  allowed values; domain failures use the flat classes in `pyfixest/errors/`.
- Guard optional dependencies at import time and raise an actionable message
  naming the pip extra only when the optional path is used.
- Use `np.random.default_rng(seed)`, never global seeding.
- Never mutate user input except the documented `copy_data=False` path.
- Post-estimation code that needs stripped data must fail informatively under
  `store_data=False` or `lean=True`.
- Every estimation/inference feature defines and tests behavior for weights,
  fixed effects, IV, and multiple estimation, or raises a specific unsupported
  error. Silent wrong results on those paths are the highest review concern.

Public functions, methods, and classes need NumPy docstrings. User-facing
entries include complete Parameters/Returns, an executable `{python}` example,
root-relative `.qmd` links, and a linked paper for econometric methods.

## Numerical references are mandatory

Every new estimator must be tested permanently against existing software. This
rule always applies; simulation properties, shape checks, and internal
reimplementations are additional evidence, not substitutes. Prefer:

1. R `fixest` or another established R package through rpy2;
2. stored output from Stata or another established implementation, including
   the generator script;
3. another external package with its exact version recorded.

If no external implementation is available, the estimator is not merge-ready.
Numerical changes to existing estimators also require an external comparison
where overlapping software exists. Record explicit tolerances and why they are
appropriate. Numerical assertions must identify the quantity being compared
and use a tolerance chosen for that quantity; do not reuse a looser inference
or prediction tolerance for coefficients. Follow `tests/test_vs_fixest.py` for
`feols`/`fepois`/`feglm` comparisons and `tests/test_quantreg.py` for
`quantreg`; `docs/developer/testing.md` defines the error criterion.

R tests use strict markers: `against_r_core` for conda-forge dependencies and
`against_r_extended` for CRAN-only extras. Add every new rpy2-importing test
file to `_rpy2_test_files` in `tests/conftest.py`. Extend existing
parametrized public-API matrices instead of creating many narrow wrapper tests.

## Verification and documentation

Use `pixi run` for every Python, pytest, lint, docs, and R command. Bare tools
may miss dependencies or the compiled extension.

- Edit loop: targeted tests with `-x -q --no-cov` and changed-file lint/type
  checks.
- PR baseline: `pixi run test-py` plus relevant lint/type checks.
- Domain suites: R, HAC, no-JIT, docs, plots, Rust, or extended tests selected
  by the changed subsystem.
- Full available suite: `test-all`; exhaustive evidence also requires the
  CRAN-only R dependencies, applicable platform CI, and relevant benchmarks.

Long suites can take tens of minutes or more. Run narrow checks while editing,
then required domain suites once the change stabilizes. Report every applicable
check as passed, failed, deferred, or not run; never imply that a deferred check
passed.

Always update `docs/changelog.qmd`. Documentation ships with the feature. New
public functions/classes require quartodoc registration; user workflows usually
need a `docs/how-to/` guide or an extension to the nearest existing guide.
Never hand-edit generated `docs/reference/**`.

## Git, stacked PRs, and review

- Never commit to `master`; use conventional `feat/`, `fix/`, `test/`,
  `ci/`, `docs/`, or `chore/` branch names. Do not use a `codex/` prefix.
- Prefer a GitHub stacked PR when work has two or more independently reviewable
  layers. Split by dependency and reviewer concern, not file count. Keep small,
  cohesive changes in one PR and unrelated work out of the stack.
- Every stack layer must be coherent, testable against its immediate parent,
  and reviewed by a human maintainer.
- Before opening a PR, ask the user explicitly whether the proposed history
  rewrite is approved. General permission to implement or open a PR is not
  rewrite approval. State the exact branches, bases, pushed/review state,
  dependent branches, and commands before asking.
- With approval, rewrite only agent-owned local history into concise,
  self-contained commits. Every commit must address one reviewer concern, pair
  tests with behavior, pass its applicable checks, and remain small enough for
  independent human review. Remove WIP, fixup, formatting-only, and accidental
  commits.
- Never rewrite a contributor-owned branch. After review starts, do not rewrite
  silently; obtain maintainer approval and use stack-aware force-with-lease.
- Commit subjects use a conventional prefix and a short imperative description.
  Bodies explain non-obvious why, not mechanics visible in the diff.
- Agents prepare draft PRs and respond to review. Agents never merge their own
  work or invoke `gh stack merge`.

Human maintainer review is required before every merge, including every layer
of a stack. Automated review and green CI supplement this gate; they do not
replace it.

## Commands

```bash
pixi run test-py
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov
pixi run test-r-core
pixi run test-r-extended
pixi run test-r-fixest
pixi run -e py312-r test-r-fixest-fast
pixi run test-r-hac
pixi run test-all

pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check --files <changed files>
pixi run -e lint prek run mypy --files <changed files>
pixi run lint

pixi run docs-build
pixi run docs-render
pixi task list
```

Rust sources rebuild through maturin-import-hook. If the extension fails after
a Rust edit, run
`pixi run -e py312-r python scripts/setup_maturin_hook.py` once.

## Do not touch unless the task requires it

- `pixi.lock` and `Cargo.lock` except intentional dependency changes.
- `docs/_freeze/**`, generated `docs/reference/**`, `.coverage`,
  `coverage.xml`, or `docs/_site/**`.
- Unrelated user changes or files changed only by broad formatting.
