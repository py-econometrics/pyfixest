# pyfixest — guide for coding agents

pyfixest ports R's `fixest` to Python: high-dimensional fixed-effects estimation
(OLS/WLS, IV, Poisson, GLM, quantile regression), fixest formula syntax,
post-estimation tools, and Rust kernels for hot loops.

Three rules beat everything else:

1. **Mirror `fixest`** in user-facing behavior, naming, and defaults unless an
   intentional difference is documented and tested.
2. **Mirror the nearest existing implementation.** Find the in-repo precedent
   and copy its structure before writing new code.
3. **Never return silently wrong numbers.** Every estimation or inference
   feature defines and tests its behavior for weights, fixed effects, IV, and
   multiple estimation, or raises a specific unsupported error. This is the
   highest review concern.

`CLAUDE.md` is a committed thin redirect to this file; do not duplicate these
rules in tool-specific configuration.

## Where policy lives

The architecture, test policy, compatibility ledger, and Git/PR style live in
[`docs/developer/`](docs/developer/). They are authoritative. This file is a
routing layer and a repo-specific conventions list; it does not restate their
policy, and neither should skills or ad hoc prompts.

| Document | Owns |
|---|---|
| [`architecture.md`](docs/developer/architecture.md) | Core boundaries, estimation flow, extension seams |
| [`testing.md`](docs/developer/testing.md) | Runtime tiers, check-selection matrix, references, tolerances, test design |
| [`fixest-compatibility.md`](docs/developer/fixest-compatibility.md) | Intentional deviations from R `fixest` |
| [`git-and-pr-style.md`](docs/developer/git-and-pr-style.md) | Branch, commit, and PR-body conventions |

## Contributor workflow skills

The skills under `.agents/skills/` are plain files shared by every coding tool,
not tool-registered commands: when a trigger applies, read the `SKILL.md` and
follow it. A change flows plan → implement → verify → self-review → hand off,
and the table is in that order. (The user-facing analytics prompt at
`docs/skills.md` is unrelated.)

| Skill | Trigger |
|---|---|
| [`implementation-strategy`](.agents/skills/implementation-strategy/SKILL.md) | Before estimator, public API, inference, formula, or shared-core work |
| [`change-verification`](.agents/skills/change-verification/SKILL.md) | After implementation stabilizes, before handing off any code, tests, docs, CI, or metadata change |
| [`pr-review`](.agents/skills/pr-review/SKILL.md) | Final self-review before handoff, and any explicit PR review |
| [`pr-handoff`](.agents/skills/pr-handoff/SKILL.md) | Curating agent-owned commits and submitting the draft PR |

## Architecture in one paragraph

Keep the shared estimation core narrow and stable: formula parsing, model-matrix
construction, demeaning, generic fit and inference primitives, result
interfaces, and backend kernels. New estimators start as standalone add-on
functions in their own API or domain modules and compose those primitives. Do
not add estimator-specific switches to generic runners or grow model classes
with numerical logic. See `docs/developer/architecture.md` for the estimation
flow, the stable-core contract, and the extension-seam table.

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
| `docs/` | Quarto user documentation; `docs/developer/` is contributor policy and is not rendered |

## Wiring recipes

- **Estimator:** a standalone module under `estimation/api/` (or the relevant
  domain package), with its own tests, result contract, exports, and quartodoc
  registration. Keep its special cases out of the generic fit pipeline.
- **Post-estimation feature:** numerical logic in `post_estimation/`; a thin
  method on `Feols` and siblings where applicable. Template:
  `post_estimation/ritest.py` and `Feols.ritest`.
- **Vcov type:** literal in `internals/literals.py`, validation in the model,
  small dispatch method, math in `internals/vcov_utils.py` or Rust, and wiring
  through `FixestMulti`/quantreg where supported. Template: NW/DK HAC.
- **Estimation-time option:** shared typed option alias in
  `internals/literals.py`, validated at the API boundary, threaded through
  `EstimationConfig` and `plan_._build_model_kwargs`.
- **Rust kernel:** `src/<topic>.rs`, registration in `src/lib.rs`, typed stub
  in `core/_core_impl.pyi`, and a clean wrapper in `core/`. Keep a NumPy
  reference implementation when feasible. Template: `src/nw.rs` → `core/nw.py`.

Reuse formula handling, `capture_context`, `_narwhals_to_pandas`, cluster
preparation, `run_crv_loop`, and `_create_rng`; do not rederive them.

## Code conventions

- Write for an econometrics practitioner. Use econometrically meaningful names
  such as `scores`, `meat`, `bread`, `u_hat`, and `clustid`; follow the paper's
  notation where it makes the implementation easier to recognize. Cite the
  paper in the implementing function.
- Methods orchestrate; numerical computation happens in standalone functions
  operating on arrays. Return small typed result dataclasses rather than tuples
  or dicts.
- Keep functions short and single-purpose. Solver iteration loops are the main
  exception when splitting would obscure the algorithm or hurt compilation.
- Put measured, non-vectorizable hot loops in Rust. Keep everything else clear,
  ordinary NumPy rather than speculative micro-optimization.
- Use `from __future__ import annotations`, PEP 604 unions, keyword arguments
  for internal calls, and `NDArray[np.float64]` in stubs. Put shared typed
  option aliases (`Literal`) in `pyfixest/estimation/internals/literals.py`.
- Validate at the API boundary. Bad option values raise `ValueError` with the
  allowed values; domain failures use the flat classes in `pyfixest/errors/`.
  Every new or changed error or warning path needs a triggering test; see the
  test-design rules in `docs/developer/testing.md`.
- Guard optional dependencies at import time and raise an actionable message
  naming the pip extra only when the optional path is used.
- Use `np.random.default_rng(seed)`, never global seeding.
- Never mutate user input except the documented `copy_data=False` path.
- Post-estimation code that needs stripped data must fail informatively under
  `store_data=False` or `lean=True`.

Public functions, methods, and classes need NumPy docstrings. User-facing
entries include complete Parameters/Returns, an executable `{python}` example,
root-relative `.qmd` links, and a linked paper for econometric methods.

## Evidence

Every new estimator must be tested permanently against existing software, and
numerical changes to existing estimators require an external comparison wherever
overlapping software exists. Simulation properties, shape checks, and internal
reimplementations are additional evidence, never substitutes. If no external
implementation is available, the estimator is not merge-ready.

Use `pixi run` for every Python, pytest, lint, docs, and R command; bare tools
may miss dependencies or the compiled extension. `docs/developer/testing.md`
owns reference selection, markers, tolerances, the runtime tiers, and the
selection matrix that decides which checks a change requires; the
`change-verification` skill applies it.

For internal or backend refactors that must not change results, the release
contract (`test-release-contract`) is the edit-loop gate: it replays the public
estimator matrix against a pinned pyfixest release in seconds. It is a
regression alarm, not an external correctness reference; a released pyfixest
result never substitutes for R.

Four rules that are easy to get wrong:

- `test-py` is Python-only and never establishes numerical agreement.
- `test-r-fixest-fast` is edit feedback, not merge evidence.
- Report every applicable check as passed, failed, deferred, or not run. A
  deferred or CI-only check is never a pass.
- `test-release-contract` **skips** without a recorded baseline, and so does
  `test-py`. Confirm it reports passed cases, not skipped, before citing it.

```bash
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov   # targeted
pixi run -e py312 test-release-contract                             # refactor invariance, ~15s
pixi run -e py312-r test-r-fixest-fast                              # fast live-R
pixi run test-py                                                    # Python baseline
pixi run -e lint prek run ruff-check --files <changed files>        # changed-file lint
pixi run -e py312 type-check                                        # ty, whole package
pixi task list                                                      # everything else
```

Always update `docs/changelog.qmd`. Documentation ships with the feature. New
public functions/classes require quartodoc registration; user workflows usually
need a `docs/how-to/` guide or an extension to the nearest existing guide.
Never hand-edit generated `docs/reference/**`.

## Git and review

Follow `docs/developer/git-and-pr-style.md` for branch names, commit messages,
and PR bodies; the `pr-handoff` skill covers history rewriting and submission.
Never commit to `master` or use an agent identity as a branch prefix.

Prefer a GitHub stacked PR when work has two or more independently reviewable
layers. Split by dependency and reviewer concern, not file count. Every layer
must be coherent, testable against its immediate parent, and small enough for
independent human review.

Rewriting history always requires explicit user approval for that specific
rewrite; general permission to implement or open a PR is not rewrite approval.
Never rewrite a contributor-owned branch or rewrite silently after review starts.

Agents prepare draft PRs and respond to review. Agents never merge their own
work or invoke `gh stack merge`. Human maintainer review is required before
every merge, including every layer of a stack; automated review and green CI
supplement that gate rather than replacing it.

## Do not touch unless the task requires it

- `pixi.lock` and `Cargo.lock` except intentional dependency changes.
- `docs/_freeze/**`, generated `docs/reference/**`, `.coverage`,
  `coverage.xml`, or `docs/_site/**`.
- Unrelated user changes or files changed only by broad formatting.
