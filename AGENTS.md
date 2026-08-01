# pyfixest — guide for coding agents

pyfixest ports R's `fixest` to Python: high-dimensional fixed-effects estimation
(OLS/WLS, IV, Poisson, GLM, quantile regression) with fixest formula syntax, a
post-estimation toolbox, and Rust kernels for hot loops.

Two rules beat everything else:

1. **Mirror `fixest`** in user-facing behavior, naming, and defaults, unless there
   is a documented reason not to.
2. **Mirror the nearest existing implementation.** Almost every kind of change has
   an in-repo precedent. Find it and copy its structure before writing new code.

**Known-stale precedents — do not mirror these.** Some rules below are newer
than parts of the codebase. Where they disagree, the rule wins and the old code
is a cleanup target, not a template:

| Do not copy | Do this instead |
|---|---|
| `Literal` aliases in `models/feols_.py`, `internals/vcov_.py`, `demeaners.py`, `core/demean.py` (`prediction_type` duplicates `PredictionType`) | new aliases go in `internals/literals.py`, validated with `_validate_literal_argument` |
| `assert` for input validation in `_check_vcov_input` | raise `ValueError` listing the allowed values |
| `_support_*` flags | spell new ones `_supports_*` |
| `test_errors.py`'s one-function-per-error style | extend an existing parametrized matrix |

Two workflows live in `.agents/`; pick by task:

- **`.agents/feature-pr.md`** — implement a feature or build a PR end to end.
- **`.agents/align-pr.md`** — audit an existing contributor PR against this
  guide and either report the misalignments or fix them on a local branch.

This file is the tool-neutral entry point: Claude Code (via `CLAUDE.md`),
Codex, and OpenCode read `AGENTS.md` natively; if a tool in your setup does
not, point its rules/config file here instead of duplicating content.
`CLAUDE.md` is a thin redirect (`@AGENTS.md`) and is committed to the repo.
Edit workflows only in `.agents/`.

This guide is maintained by ratchet: when maintainer feedback states a
preference this file does not encode (or contradicts), encode it as part of the
same change. **Prefer a check over prose.** If the rule is mechanically
detectable it belongs in a ruff rule or in `scripts/check_house_style.py`, and
then it does *not* also belong here — a rule stated in two places drifts. This
file stays at roughly its current length: adding a rule means deleting one,
compressing one, or moving one to a check. It is read in full at the start of
every session, so length is a running cost.

## Repo map

| Path | Contents |
|---|---|
| `pyfixest/estimation/api/` | Public entry points, one module per function: `feols`, `fepois`, `feglm`, `quantreg`; shared input checks in `api/utils.py` |
| `pyfixest/estimation/models/` | Model/result classes (`Feols`, `Feiv`, `Fepois`, `Feglm`, …); modules end in `_` (`feols_.py`) |
| `pyfixest/estimation/internals/` | Shared estimation internals: `vcov_utils.py`, `solvers.py`, `collinearity.py`, `separation.py`, `literals.py` |
| `pyfixest/estimation/post_estimation/` | Post-estimation features (`ritest`, `ccv`, `decomposition`, `prediction`, `multcomp`); model classes hold only wrapper methods free of numerics |
| `pyfixest/estimation/formula/` | Formula parsing and model-matrix construction (wraps formulaic) |
| `pyfixest/estimation/` root | `config.py` (`EstimationConfig`), `plan_.py` (`parse_formula`, `fit_one`), `runner.py`, `FixestMulti_.py` (pure container for multiple estimation); backend subpackages `numba/` and `torch/`; `deprecated/`. Root-level `feols_.py`/`feiv_.py`/`fepois_.py` are compat shims — the real classes live in `models/` |
| `pyfixest/demeaners.py` | Public demeaner configs (`BaseDemeaner`, `MapDemeaner`, `LsmrDemeaner`) behind the `demeaner=` argument; own quartodoc section |
| `pyfixest/core/` | Python wrappers and `_core_impl.pyi` type stubs for the Rust extension |
| `src/` | Rust kernels (PyO3), registered in `src/lib.rs` |
| `pyfixest/did/`, `pyfixest/report/`, `pyfixest/utils/` | DiD estimators; `etable`/plots; data utilities and DGPs (`utils/dgps.py`) |
| `tests/` | Pytest suite; reference scripts and stored outputs in `tests/data/` |
| `docs/` | Quarto site, Diataxis layout: `tutorials/`, `how-to/`, `explanation/`, `textbook-replications/` |

Estimation flow: `feols()` builds an `EstimationConfig` → `parse_formula`
(`plan_.py`) expands multiple-estimation syntax → `FixestMulti` holds the models →
`runner.run_estimation` / `plan_.fit_one` drive each model through
`prepare_model_matrix → get_fit → vcov → get_inference`, where `get_fit` runs
`demean → to_array → drop_multicol_vars → wls_transform` before solving.
Post-estimation happens via methods on the fitted model.

## Where new code goes

**New post-estimation feature** — standalone module in
`estimation/post_estimation/` holding the logic; a thin method on `Feols` (and
siblings where applicable) that validates inputs and delegates. The commit history
is one long effort to carve logic *out* of the model classes — do not grow them.
Template: `post_estimation/ritest.py` + `Feols.ritest`.

**New vcov type** — add the option to `internals/literals.py`; accept and
validate it in `_check_vcov_input` / `_deparse_vcov_input`
(`models/feols_.py`); dispatch from `Feols.vcov()` via a small `_vcov_<name>`
method; put the meat/bread math in `internals/vcov_utils.py` (or a Rust kernel);
thread through `FixestMulti.vcov()` and quantreg if applicable; wire ssc via
`_make_ssc_kwargs`. Template: the NW/DK HAC path.

**New estimation entry point** — own module in `estimation/api/`; export through
`estimation/api/__init__.py` and `pyfixest/__init__.py` (`__all__`,
`_lazy_imports`, `_direct_module_imports`); add to the quartodoc `contents` list
in `docs/_quarto.yml`. Keep the signature order consistent with siblings:
`fml, data, vcov, …, copy_data, store_data, lean, …`. User-facing functions that
are *not* estimation entry points (`rwolf`, `bonferroni`, `wyoung`) instead live
in `post_estimation/` and are exported through `estimation/__init__.py` plus the
top-level `_lazy_imports`.

**New Rust kernel** — `src/<topic>.rs` with a function named `_<name>_rs`;
register in `src/lib.rs`; add a typed stub to `pyfixest/core/_core_impl.pyi`;
re-export under a clean alias in `pyfixest/core/<topic>.py`. Keep a NumPy
reference implementation around for tests when feasible. Template: `src/nw.rs` →
`pyfixest/core/nw.py`.

**New estimation-time option** — `Literal` alias in `internals/literals.py`;
accept it in each relevant `api/` function with a documented default; validate
early (`_validate_literal_argument` or `api/utils._estimation_input_checks`);
thread through `EstimationConfig` and `plan_._build_model_kwargs`.

Reuse before you write: formula handling in `estimation/formula/`, context
capture via `utils.utils.capture_context`, dataframe conversion via
`utils.dev_utils._narwhals_to_pandas` (`DataFrameType` accepts pandas/polars/
duckdb), cluster prep via `internals/vcov_utils.prepare_cluster_state` /
`run_crv_loop`, RNG via `utils.dev_utils._create_rng`.

## House style

Write for an econometrics practitioner first. A reader who knows the method from
the paper should recognize it in the code:

- Name objects after the econometrics they represent — `scores`, `meat`,
  `bread`, `u_hat`, `clustid` — and mirror the source paper's notation where it
  has one. Matrix products use the `t` prefix for transpose: `tZX` is Z'X,
  `tZZinv` is (Z'Z)^-1. Cite the paper (with a link) in the docstring of the
  function that implements it, as `Feols.decompose` does for Gelbach (2016).
  User-facing names spell out what they do even at the cost of length
  (`dfm_heterogeneity_test`, not `dfm_test`).
- Methods orchestrate; numbers happen in functions. A model method validates
  inputs, unpacks `self._` state into locals, calls a standalone module-level
  function that operates on arrays, and assigns the results back to `self._`
  attributes. The numerical function never touches `self` — that seam is what
  makes it testable against a reference. `Feols.get_fit` →
  `internals/fit_.fit_ols` is the template. As a rule of thumb, more than ~30
  added lines in `estimation/models/` means the code is in the wrong file.
- A feature never adds an overridable hook method to the model classes.
  Per-model behavior is selected by a `_supports_*` flag, or by passing the
  variant to the function as an argument — defining `_do_x()` on `Feols` and
  overriding it in `Feiv` and `Feglm` spreads one feature's numerics across
  three files and puts them out of reach of a reference test.
- Numerical functions return a small result dataclass (`OlsFit`, `IvFit`,
  `ClusterPrep`; frozen and slotted where possible) whose Attributes docstring
  states each array's shape — not a tuple, not a dict. Internal calls pass
  arguments by keyword: `fit_ols(X=self._X, Y=self._Y, solver=self._solver)`.
  Every attribute a dataclass sets is declared as a field — `field(init=False)`
  for ones computed later; assigning `self._foo` in a method without declaring
  it hides state from readers and from mypy.
- One function, one named task, kept short — most functions in the codebase are
  under ~30 code lines. If a function's name or docstring summary needs an
  "and", split it. The fit pipeline is the model: `prepare_model_matrix →
  get_fit → vcov`, with `get_fit` stepping through `demean → to_array →
  drop_multicol_vars → wls_transform`, each step a small named unit. Sanctioned
  exceptions to "short": a single solver iteration loop (IRLS, LSMR,
  Frisch–Newton — splitting it hurts readability and `torch.compile`) and
  validation-heavy user-facing methods. Split logic, not loops. The inverse
  also holds: a public function that only forwards to a same-signature private
  `_twin` is one function too many.
- Performance-critical code — per-observation or per-cluster hot loops that
  NumPy cannot vectorize — is written and optimized in Rust under `src/` (the
  demean, CRV1-meat, and HAC-meat kernels are the pattern), not micro-optimized
  in Python. Everything else stays plain, readable NumPy: do not trade clarity
  for speed outside a measured hot loop, and keep a NumPy reference
  implementation around for testing the kernel. A contributed numba kernel
  becomes a Rust kernel, with the plain-NumPy version kept as the test
  reference.

Naming and layout:
- One public function per `api/` module; model-class modules end in `_`
  (`feols_.py` holds `Feols`) so they don't shadow the API function names. New
  helper modules are plain snake_case.
- Private helpers are module-level `_underscore` functions kept in the same
  module as their only caller, or in `internals/` when shared.
- Model state is `self._underscore` (`self._data`, `self._weights`,
  `self._is_iv`); results are exposed through methods (`tidy()`, `coef()`,
  `se()`) — shared accessors live in `models/_result_accessor_mixin.py`.

Signatures and typing:
- `from __future__ import annotations`; PEP 604 unions (`str | None`); typed
  option enums as `Literal` aliases in `internals/literals.py`.
- In docstring Parameters sections, spell types as in the signature
  (`str | None`, not `Optional[str]`). Older docstrings mix spellings — don't
  copy them.
- Public estimation API: `fml` first, `data` second, typed `DataFrameType`
  (accepts pandas/polars/duckdb) and converted once at the boundary via
  `_narwhals_to_pandas`.
- mypy runs on `pyfixest/` only; `NDArray[np.float64]` in the `.pyi` stubs.

Docstrings (ruff enforces the NumPy convention):
- Required on public functions/methods/classes; not required in `tests/`.
- User-facing API docstrings carry full Parameters/Returns and an Examples
  section with executable ```{python} chunks — quartodoc renders and runs them.
- Cross-link classes as `[Feols](/reference/estimation.models.feols_.Feols.qmd)`
  and vignettes as `[guide](/tutorials/standard-errors.qmd)` — always
  root-relative with a `.qmd` extension. A few older docstrings use relative
  paths or `.html`; don't copy them.
- Inline code references use single backticks.

Errors, warnings, optional dependencies:
- Validate at the API boundary and fail fast. Bad option values raise
  `ValueError` listing the allowed values; domain errors use the flat exception
  classes in `pyfixest/errors/__init__.py` (add one there if none fits).
- Deprecate arguments with `warnings.warn(..., UserWarning)` while keeping the
  old spelling working (see `Feols.decompose`'s `param` → `decomp_var`).
- Optional deps (`numba`, `lets_plot`, `torch`) are guarded with
  `try/except ImportError` and a `_HAS_X` flag at module top; raise an
  actionable message naming the pip extra only when the path is actually used
  (see `post_estimation/ritest.py`).

Numerics and data:
- RNG is always `np.random.default_rng(seed)` — one generator per test or DGP.
  ruff `NPY002` enforces this; do not extend its grandfather list in
  `pyproject.toml`.
- Never mutate user input data. `copy_data=False` is the only sanctioned
  exception, and its side effects are spelled out in the docstring.
- `store_data=False` and `lean=True` strip attributes
  (`Feols._clear_attributes`): post-estimation code that needs `self._data`
  must raise an informative error, not crash.
- Every estimation/inference feature defines its behavior under weights
  (`aweights` vs `fweights`), fixed effects, IV (`Feiv`), and multiple
  estimation (`FixestMulti`) — test the supported paths, raise
  (`NotImplementedError`, `VcovTypeNotSupportedError`) on the rest. Silent
  wrong numbers on these paths are the number-one review concern.

Comments are sparse and state constraints the code can't (the rpy2 converter
note in `tests/conftest.py`, the lazy-numba note in `ritest.py`) — no narration.

## Testing

- Quick suite: `pixi run test-py`. Targeted:
  `pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov`
  (`--no-cov` skips the coverage report that pytest addopts force on every run).
- Markers (strict): `extended`, `against_r_core` (conda R packages),
  `against_r_extended` (CRAN extras), `plots`, `hac`. The quick suite excludes
  all of them.
- Any new test file importing `rpy2` must be added to `_rpy2_test_files` in
  `tests/conftest.py` so non-R environments skip it.
- Prefer a few heavily parametrized *integration* tests that drive the public API
  (`feols`/`fepois`/`feglm`/`quantreg`) end to end and check against a reference,
  over many small unit tests. `tests/test_vs_fixest.py` is the archetype — one
  parametrized matrix over formulas × vcov × weights × ssc, all compared to R
  `fixest`. Reserve unit tests for internal seams that are awkward to reach through
  the API: the demean kernel (`test_demean.py`), formula parser
  (`test_formula_parse.py`), HAC meat (`test_hac_meat.py`) — not thin wrappers or
  getters.
- When a change fits an existing parametrized matrix, extend it — add a formula
  to the module-level list or a case to the matrix — rather than writing a new
  test function.
- At least one test per feature calls `feols`/`fepois`/`feglm`/`quantreg` and
  drives the feature through it. The wiring *is* the feature: a module tested
  only in isolation can ship with a public entry point that raises. Where the PR
  description shows a usage example, that exact call is a test.
- Style: module-level `test_*` functions, no banner comments; shared setup is a
  seeded DGP fixture (module-scoped when expensive). Module-level formula lists
  fed to `pytest.mark.parametrize`; explicit `rtol`/`atol` constants at the top
  of the file with a comment justifying them.
- The bar for new econometrics is higher than "runs and has the right shape":
  exact known-value or brute-force cross-checks, edge cases (singleton
  clusters, collinearity, tiny samples), invalid-input tests, and at least one
  external reference — R `fixest`/`sandwich` via rpy2, stored Stata output
  committed under `tests/data/`, or a simulation property (empirical
  size/coverage). Every file under `tests/data/` ships with the script that
  generated it (`.do`, `.R`, or `.py`) in the same directory; hard-coded
  reference numbers in a test cite their generator the same way. Code adapted
  from elsewhere gets a provenance and license note in its docstring (see
  `tests/test_ccv.py`).
- Reference packages: on conda-forge → add as a dependency (R packages →
  `[tool.pixi.feature.r.dependencies]`) and mark the test `against_r_core`, so
  CI runs the comparison live. Not on conda-forge → either CRAN via
  `r_test_requirements.R` with the test marked `against_r_extended` (local only),
  or commit a generator under `tests/data/` and hard-code its values. Both are
  fine; reach for hard-coded values over a heavy or flaky live dependency.

## Docs

Docs ship in the same PR as the feature — a reviewer treats missing docs as an
incomplete change. What to touch depends on the surface:

- New estimation-time option or `vcov` value: `feols`, `fepois`, `feglm`, and
  `quantreg` each carry their *own* Parameters docstring (there is no shared
  docstring), so document the option in every one that accepts it, and in the
  relevant model method (e.g. `Feols.vcov`).
- New post-estimation method: a full NumPy docstring with an executable Examples
  section (a `{python}` chunk) on the method itself — quartodoc renders and runs
  it into the reference.
- New class or function: register it in the quartodoc `contents`
  (`docs/_quarto.yml`); `pixi run docs-build` regenerates `docs/_sidebar.yml` and
  the reference pages.
- Vignette: add a `docs/how-to/<feature>.qmd` (hyphenated filename) when the
  feature is a workflow a user would want a guide for, and register it in the
  `_quarto.yml` navbar; extend the nearest existing vignette instead when the
  feature only widens one (a new vcov type → the standard-errors guide,
  `docs/tutorials/standard-errors.qmd`). The Conley and decomposition features
  are the model.
- Every user-facing change adds an entry to `docs/changelog.qmd` under the
  current "(In Development)" release heading, in the style of the entries
  already there.
- Never hand-edit `docs/reference/**` — it is generated by quartodoc and
  gitignored. Reference display names are shortened by a custom renderer
  (`docs/_renderer.py`); the `docs-build` task runs quartodoc from `docs/` so it
  can import it.

Every entry in the quartodoc reference (`contents` in `docs/_quarto.yml`) meets
the same bar, whether it is a function, a method or a class:

- Description: say what the object does and when to use it, not just what it
  returns. A one-line summary is not enough for a user-facing entry.
- At least one executable example: an `Examples` section with a `{python}` chunk,
  which quartodoc runs into the page. Getters and accessors (`coef()`, `se()`, a
  `get_*` generator) included — two lines are enough. Keep examples fast (small
  `pf.get_data()` samples, few `reps`); skip one only when runtime is
  prohibitive, and say so.
- Vignette link when a relevant guide exists, as
  `[guide](/how-to/<feature>.qmd)`. Inference methods point at the
  standard-errors guide (`/tutorials/standard-errors.qmd`), `decompose` at the
  decomposition vignette.
- Paper citation with a link whenever an econometric method is implemented, in
  the function that implements it. `Feols.decompose` cites Gelbach (2016),
  `quantreg` cites Portnoy and Koenker (1997). Mirror the paper's notation in
  the code (see "House style").

Result-class examples (`Feols`, `Fepois`, `Feiv`, …) fit a model and call a
method rather than constructing the class. Examples go straight to the
`{python}` chunk with any explanation in short comments inside the code — do not
narrate in prose (the long narrated Examples in `feols`/`fepois` predate this
rule). Verify an example runs before handing off: at minimum execute its body in
`pixi run python`; the full check is `pixi run docs-build` then
`QUARTO_PYTHON=.pixi/envs/docs/bin/python3 quarto render docs/reference/<page>.qmd`
(quarto ignores the `python:` key for single-file renders).

## Commands

```bash
pixi run test-py                                   # quick suite, no R
pixi run -e py312-r pytest tests/test_x.py -x -q --no-cov   # targeted, R available
pixi run -e py312-r pytest tests -m "not (extended or against_r_core or against_r_extended or plots or hac)"
pixi run test-r-core                               # R comparisons, conda-forge packages
pixi run test-r-extended                           # R comparisons, CRAN extras (local only)
pixi run test-r-fixest                             # tests/test_vs_fixest.py only
pixi run test-r-hac                                # HAC vs R, single-threaded BLAS
pixi run test-all                                  # everything

pixi run -e lint prek run ruff-format --files <changed files>
pixi run -e lint prek run ruff-check  --files <changed files>
pixi run -e lint prek run mypy       --files <changed files>
pixi run lint                                      # all hooks, all files
pixi run check-style                               # house-style checks over the branch diff

pixi run docs-render                               # full docs (runs quartodoc docs-build first)
pixi task list                                     # everything else
```

Always go through `pixi run`; bare `python`/`pytest` may lack dependencies or
the compiled Rust extension. Rust sources rebuild automatically on import via
maturin-import-hook; if `pyfixest.core._core_impl` fails to import after
touching `src/`, run `pixi run -e py312-r python scripts/setup_maturin_hook.py`
once.

## Do not touch (unless the task is about them)

- `pixi.lock`, `Cargo.lock` — only for intentional dependency changes, in their
  own commit.
- `docs/_freeze/**` — frozen notebook output; re-render only pages your change
  affects.
- `docs/reference/**` — generated by quartodoc and gitignored; never hand-edit.
- `.coverage`, `coverage.xml`, `docs/_site/**` — local artifacts.
- Do not reformat files you didn't otherwise change; keep diffs reviewable.

## Git and PRs

- Never commit to `master` (a pre-commit hook blocks it); branch first.
- Run lint and type checks on the changed files before each commit: `ruff-format`,
  then `ruff-check`, then `mypy` (commands under "Commands"). The same hooks gate
  the PR in CI, so a red check blocks the merge. `ruff-format` and `ruff-check
  --fix` rewrite files in place — stage the result before committing. These fire
  automatically only if `prek install` set up the git hook, which agents and fresh
  clones usually haven't done, so run them explicitly rather than relying on it.
  mypy is scoped to `pyfixest/`, not `tests/`.
- Commit subjects: a conventional prefix plus one short, precise, imperative
  line that says what changed — e.g. `feat: add transform building blocks`,
  `refactor: make FixestMulti a pure container`, `chore: remove dead code from
  Feols`. Use `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`,
  `build`, `ci`; add a scope when it narrows usefully (`fix(vcov): ...`).
  Target ~50–60 chars including the prefix, no trailing period, backtick code
  identifiers. Don't hand-append `(#PR)`; GitHub adds it on squash-merge.
- Commit bodies: add one only when the *why* isn't obvious from the subject,
  and keep it to a line or two. Write plain English that a contributor reading
  `git log` in a year can follow without opening the diff. Name the thing, then
  say what is true of it, in the present tense and in ordinary words:
  - "`_feols_input_checks` is never called" — not "has no callers".
  - "`na_index` is only assigned, never read" — not "was only ever written".
  - "the second `_supports_wildboottest = True` overwrites the first, so the
    weights check above it never takes effect" — not "the guard was a no-op".

  Prefer a longer plain sentence over a short cryptic one. If a phrase would
  need a follow-up question to explain it, it is the wrong phrase.
- Before handing off a PR, rewrite the branch into a few logically ordered,
  individually test-passing commits (helpers + tests → wiring + tests → API
  exposure + docs). The maintainer reviews commit by commit; see
  `.agents/feature-pr.md`, Phase 5, for the guarded rewrite recipe.
