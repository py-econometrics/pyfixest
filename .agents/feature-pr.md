# Workflow: implement or clean up a pyfixest feature PR

Single, tool-neutral source of truth for feature-PR work, reached through the
pointer in `AGENTS.md`. Claude Code, Codex, and OpenCode read `AGENTS.md` natively;
if a tool in your setup does not, point its rules/config file at these two
documents instead of duplicating content. Edit the workflow in this file only.

Read `AGENTS.md` at the repo root first: it defines the repo map, wiring
checklists, house style, and commands this workflow assumes. This file defines the
process. Stay inside its bounded loops — no unbounded retrying.

Inputs: a feature issue, or an existing PR to clean (`gh pr checkout <n>`,
`gh pr diff <n>`). Always work on a feature branch; a pre-commit hook blocks
commits to `master`.

## Phase 1 — Discovery (run once)

1. Establish the diff base once; reuse it in every later phase. The PR's base
   branch beats local `master`, which may be stale. Run `git fetch origin` when
   network/auth permits; if it fails, continue with the existing remote-tracking
   ref and say in the handoff that the base may be stale.

   Resolve the base branch (`gh pr view --json baseRefName -q .baseRefName` for
   PR cleanup; otherwise `master` unless the issue names another base), then
   compute and record the literal values in the plan because agent shells may not
   share environment variables:

   ```bash
   BASE_REF=origin/<base-branch>
   MERGE_BASE=$(git merge-base HEAD "$BASE_REF")
   ```

2. Collect the change surface. PR cleanup: `gh pr diff <n> --name-only` and the
   PR description. New feature: the issue plus the subsystems you expect to
   touch.
3. Map every file to a subsystem: `estimation/api` | `estimation/models` |
   `estimation/internals` | `estimation/post_estimation` | `estimation/formula` |
   `core` + `src` (Rust) | `tests` | `docs`.
4. Classify each diff hunk into exactly one bucket:
   - feature logic
   - tests
   - docs
   - formatting-only churn → drop
   - generated artifacts (`pixi.lock`, `Cargo.lock`, `docs/_freeze/**`,
     `coverage.xml`) → drop unless the task is explicitly about dependencies
   - unrelated refactor → drop, note it for a follow-up issue
5. Read the nearest analogue end to end before writing code:
   - post-estimation method → `estimation/post_estimation/ritest.py` and the
     `Feols.ritest` wrapper
   - vcov type → the HAC path: `internals/vcov_utils.py`, `Feols._vcov_hac`,
     `_deparse_vcov_input`
   - new API function → `estimation/api/quantreg.py` and the export chain in
     `pyfixest/__init__.py`
   - Rust kernel → `src/nw.rs` → `src/lib.rs` → `pyfixest/core/_core_impl.pyi`
     → `pyfixest/core/nw.py`
   - reference test → `tests/test_hac_vs_fixest.py`
6. Write a plan of at most ten lines: `BASE_REF=<literal>`,
   `MERGE_BASE=<sha>`, target files, wiring points, the numerical reference you
   will validate against (Phase 3), and the churn you will remove. Then start.

## Phase 2 — Implementation loop (max 3 iterations per slice)

Slice the work so each slice builds and tests on its own:

1. helper module in `post_estimation/` or `internals/` + unit tests,
2. model/dispatch wiring (thin method, validation) + integration tests,
3. public API exposure + docs.

Per iteration: make the smallest coherent change, then run the narrowest tests
that cover it:

```bash
pixi run -e py312-r pytest tests/test_<feature>.py -x -q --no-cov
```

A feature rarely lives in one new test file — also run the existing files for
the touched subsystem (e.g. changes to `vcov_utils.py` implicate
`tests/test_ses.py` and `tests/test_crv1_vcov.py`; prediction code implicates
`tests/test_predict_resid_fixef.py`). `--no-cov` skips the coverage report that
pytest addopts otherwise force on every run; coverage comes with the broader
runs in Phase 4. Fix what the traceback says. Three failed iterations on the
same slice → Escalation.

While implementing, enforce the AGENTS.md rules that PRs most often break:
- Bulky logic never lands in model classes; the class method validates,
  delegates, and carries the NumPy docstring.
- Reuse `estimation/formula/`, `_narwhals_to_pandas`, `capture_context`,
  `prepare_cluster_state`/`run_crv_loop`, `_validate_literal_argument` — do not
  re-derive parsing, conversion, or cluster prep.
- New option names get `Literal` aliases in `internals/literals.py` and early
  validation.
- Never mutate user data; handle `store_data=False` / `lean=True` with an
  informative error.
- Write for an econometrics reader: paper notation (`scores`, `meat`, `bread`),
  the paper cited in the docstring, one short named task per function. Hot
  loops that NumPy cannot vectorize go to a Rust kernel in `src/`; everything
  else stays plain NumPy (AGENTS.md → "House style").

## Phase 3 — Numerical validation loop (before claiming success)

An econometric feature is not done because shapes match. Validate against the
strongest reference available, in this order of preference:

1. R (`fixest`, `sandwich`, `quantreg`, …) via rpy2 — mark `against_r_core`
   (conda packages) or `against_r_extended` (CRAN extras); add any new
   rpy2-importing test file to `_rpy2_test_files` in `tests/conftest.py`.
2. Stored Stata or otherwise verified output committed under `tests/data/`,
   including the script that produced it.
3. A brute-force / dense reimplementation inside the test.
4. A closed-form special case that must collapse to an existing estimator
   (lag-0 HAC equals HC; equal weights equal unweighted; a single fixed effect
   equals explicit dummies).
5. Seeded Monte Carlo properties (`np.random.default_rng(seed)`): empirical
   size near nominal under the null, power against a fixed alternative.

Dependency rule for reference packages (full version: AGENTS.md → "Testing"):
on conda-forge → add as a dependency and mark `against_r_core` (runs in CI);
not on conda-forge → CRAN via `r_test_requirements.R` (repo root) with the test
marked `against_r_extended` (runs locally only), or commit a generator script
and hard-code its values into the test. Both are acceptable.

Prefer a few heavily parametrized integration tests through the public API — the
`tests/test_vs_fixest.py` shape (formulas × vcov × weights × ssc vs R `fixest`) —
over many isolated unit tests; keep unit tests for internal seams the API can't
reach (demean kernel, formula parser, HAC meat). Cover the paths the feature
claims to support: no fixed effects vs fixed effects, unweighted vs weighted,
clustered, IV, and multiple estimation.
Assert invariants where cheap: vcov symmetric with positive diagonal,
row-order invariance. Every `rtol`/`atol` gets a one-line justification; do not
loosen a tolerance to make a test pass without explaining why the looser bound
is expected.

## Phase 4 — Review loop (at most 2 passes)

Recompute or restore `BASE_REF` and `MERGE_BASE` from the Phase 1 plan if the
shell session changed. Self-review everything that would ship, not only committed
changes:

```bash
git status --short
git diff "$MERGE_BASE"
git diff --cached "$MERGE_BASE"
```

`git diff "$BASE_REF"...HEAD` is useful for reviewing committed branch history,
but it misses uncommitted and untracked work; `git status --short` is the guard
for those. Review the full change against AGENTS.md:

1. Any logic sitting in a model class that belongs in `post_estimation/` or
   `internals/`? Move it; leave a thin wrapper.
2. Any formatting churn, generated artifacts, lockfile or dependency churn,
   or unrelated refactors left in the diff? Remove them.
3. Public surface consistent with `feols`/`fepois`/`feglm`/`quantreg` and
   existing post-estimation methods: argument names and order, defaults,
   NumPy docstring with Examples, return types (`pd.DataFrame`/`pd.Series` or
   a results object with `tidy()`).
4. Exports and docs go last, once implementation and tests are coherent (see
   AGENTS.md → "Docs"): `pyfixest/__init__.py` (`__all__`, `_lazy_imports`); the
   Parameters docstring in each of `feols`/`fepois`/`feglm`/`quantreg` that gained
   an option (they do not share docstrings); quartodoc `contents` in
   `docs/_quarto.yml` for a new class or function; and a `docs/how-to/*.qmd`
   vignette (navbar entry in `_quarto.yml`) when the feature warrants a guide.
   For every public change, update canonical docstrings/authored docs first and
   then regenerate `pyfixest/docs/` with
   `pixi run -e py312-r python docs/_scripts/generate_llms_md.py --no-site`.
   Never hand-edit generated package corpus files, `docs/skills.md`, or
   `docs/llms.txt`. Update the canonical `skills/pyfixest/` skill whenever it
   changes agent routing or a documented workflow, and validate it with
   `scripts/check_skill.py`.
5. Run in order: targeted tests → the three lint hooks on changed files, one at
   a time — `pixi run -e lint prek run ruff-format --files <changed>`, then the
   same command with `ruff-check`, then with `mypy` → `pixi run test-py` if
   shared internals (`vcov_utils`, `plan_`, model classes, formula code) were
   touched.

## Phase 5 — Commit history rewrite (before handing off)

Recompute or restore `MERGE_BASE` from the Phase 1 plan if the shell session
changed. Inspect `git log --oneline "$MERGE_BASE"..HEAD`. If it is not a short
sequence of coherent commits, rewrite it. Interactive `git rebase -i` is
unavailable in some agent sandboxes; the soft-reset recipe below works — but it
collapses all committed work onto the index, so verify every precondition first:

1. `git rev-parse --abbrev-ref HEAD` names a feature branch, not `master`.
2. `git status --short` is clean. A soft reset silently mixes uncommitted edits
   into the re-committed slices — if anything is dirty, stop and resolve it
   (commit, stash, or ask the user) before rewriting.
3. The branch is yours to rewrite: unpushed, or the user has approved a force
   push. Never rewrite a contributor's branch without flagging it in the
   handoff summary.
4. `MERGE_BASE` is set (recompute it now if the shell session changed) and an
   escape hatch is recorded, so a botched re-slicing is recoverable with
   `git reset --hard "$TIP"`.

```bash
TIP=$(git rev-parse HEAD)
git reset --soft "$MERGE_BASE"
# re-commit in ordered slices (git add <paths> per slice)
```

Target shape, two to five commits mirroring the slices in Phase 2: helpers +
tests → wiring + tests → docs/exports. Subjects are short, precise, and
imperative (~50–60 chars, no trailing period; see AGENTS.md → "Git and PRs" for
the full format); conventional prefixes are optional. Each commit should pass
its own targeted tests. Do not rewrite commits that are already pushed to
someone else's branch without saying so in the handoff summary.

## Escalation rule

Stop and report instead of widening the change when:
- the same failure repeats three times,
- the fix seems to require touching lockfiles, `docs/_freeze/**`, or generated
  files, or
- correctness against the reference cannot be established.

The report names the exact files, the exact commands run, the observed output,
and your best hypothesis — nothing vaguer.

## Command reference

All commands are defined once, in AGENTS.md → "Commands" — look them up there
rather than copying them here. The only workflow-specific form is the Phase 2
inner loop:

```bash
pixi run -e py312-r pytest <test files> -x -q --no-cov
```
