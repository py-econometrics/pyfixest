# PyFixest: test-suite review and a development-harness sketch

Companion to `codebase_review_fable.md`. Part 1 assesses the test suite; Part 2 sketches a development harness — how the codebase wants to be written in, and how to make contributors (human and agent) write in that style with a fast feedback loop. Numbers come from running the suite in a sandbox on the `refactor/weights-wiring` branch.

---

## Part 1 — The test suite

### Snapshot

51 test files, ~13,700 lines. Roughly 1,500 tests collect without R; ten files (led by `test_vs_fixest.py` at 1,848 lines) additionally require rpy2/R and are cleanly skipped via `conftest.collect_ignore` when it's absent. Five markers (`against_r_core`, `against_r_extended`, `extended`, `plots`, `hac`) are declared with `--strict-markers` and wired into pixi tasks (`test-py`, `test-py-extended`, `test-r-core`, `test-r-extended`, `test-r-fixest`, `test-r-hac`) that CI runs as separate jobs with partial-coverage merge. The conftest is thoughtful: an rpy2 converter that preserves R list classes, single-threaded BLAS forced for HAC determinism (with a comment explaining the IEEE-754 reasoning), and a pre-xdist warm import to avoid Rust build races.

### What is working well

The single best decision in this suite is **differential testing against R fixest as the oracle**. For an econometrics library there is no stronger correctness signal than "agrees with the reference implementation to 1e-8 across coef, se, t-stats, p-values, CIs, residuals, predictions, and R²" — hand-computed golden values could never cover this surface. The second strength is the growing set of **internal-identity tests that need no oracle**: `test_ses.py` checks that HC1 equals CRV1 clustered on the observation ID (a mathematical identity), `test_feols_feglm_internally.py` checks feols against the gaussian GLM, the demeaner tests check Rust against numba backends, and `test_weights_wiring.py` (new) checks that fweights equal physical row duplication. These are fast, deterministic, and localize failures well. Third, the **Rust kernels each have direct unit tests** (demean, collinearity detection, CRV1 meat, HAC meat, singleton detection, nested-FE counting), so the numerical core is not only tested through the API. Fourth, `test_errors.py` (1,074 lines, 45 tests) gives error-path coverage most scientific libraries skip. The marker/pixi/CI architecture is coherent and documented in `docs/contributing.qmd`.

### Weaknesses

**The pyramid is top-heavy, and failure localization suffers.** Most correctness weight sits in `test_vs_fixest.py`'s Cartesian grids. `test_single_fit_feols` alone multiplies ~48 formulas × 3 inference types × 2 weights × 2 dropna × ~9 backend/dtype combos — thousands of full Python-plus-R fits. When something in `get_ssc` regresses, the signal is hundreds of red parametrized cases across four test functions, not one failing unit test that names the culprit. The grid is the right *nightly* artifact; it is a blunt *per-PR* instrument.

**Orthogonal dimensions are multiplied instead of composed.** The `f3_type` dimension (str/object/int/categorical/float for one FE column) tests dtype coercion; it is independent of which formula or inference type is used, yet it multiplies the whole grid. Same for demeaner backends: backend equivalence is a property of the demeaner (already unit-tested in `test_demean.py`), not something that needs re-verification per formula × inference cell. A pairwise-style design — full formula list on one base configuration, plus each non-default dimension varied against the base — would cut R-grid runtime by an estimated 5–10x with nearly identical fault-detection power.

**The "quick" tier is not quick.** `pixi run test-py` selects ~650 tests, but single-threaded it runs for many minutes (it did not finish within 5 minutes in my sandbox; CI leans on 4 xdist workers). The bulk sits in a handful of files: `test_ses.py` (108 configs × several fits each), `test_ritest.py` (randomization loops that refit hundreds of models on the no-numba path), `test_decomposition.py` (bootstraps). Meanwhile a genuinely fast core exists — `test_weights_wiring` + `test_solvers` + `test_multicollinearity` + `test_exceptions` + `test_plan` + `test_formula_parse` is 162 tests in **2 seconds** — but there is no named tier that selects it, so nobody (and no agent) can use it as an inner loop.

**The seam layer between API and kernels is under-unit-tested, and that is exactly where the bugs were.** Every confirmed bug in `codebase_review_fable.md` lived in pure-Python glue with no direct test: `_deparse_vcov_input` (no unit test → the list-vcov `UnboundLocalError` and the dead `"^"` branch survived indefinitely), the GLM weights plumbing (fweights+fepois crashed — note R fixest has no fweights, so the R oracle *structurally cannot* cover this feature; it had neither oracle nor invariant test until this branch), `wald_test`'s distribution-switching, `update()` under weights, `Feglm.predict`'s swallowed arguments. `fit_ols`/`fit_iv`, `vcov_hetero`/`vcov_iid_ols`, and the prediction helpers are likewise only exercised end-to-end.

**Thin coverage of the configuration surface.** `lean=True` has no contract test (which methods still work, which raise, and how gracefully); `copy_data=False`'s mutation behavior is asserted nowhere; there is no test that multiple-estimation syntax (`sw()`/`csw()`, i.e. the demean cache) produces results identical to the equivalent separate fits — the cache is pure risk with no dedicated correctness test.

**Small mechanics.** `check_absolute_diff` is copy-pasted into three files (`test_vs_fixest`, `test_hac_vs_fixest`, `test_feols_feglm_internally`) with slightly diverging behavior — it belongs in a `tests/utils.py` together with the data factories and the row-expansion helper from `test_weights_wiring.py`. Tolerances are ad-hoc per test site; a small central tolerance policy (documented constants per comparison type) would make deviations reviewable.

### Are we over-testing? Should we shift from integration to unit tests?

Over-testing: only at the margins of the R grids, in the multiplied-orthogonal-dimensions sense above — the cost is CI time and failure noise, not wrongness. Do not reduce what the grid *covers*; reduce how often the full product runs (nightly) versus the curated slice (per PR).

Integration → unit: the answer is **add under, don't move away**. The R-differential tests are the crown jewels and the reason users trust the library; converting them to unit tests would destroy information. The gap is at the bottom of the pyramid, and the recent bug harvest tells you precisely where. A workable going-forward rule: every internal function that branches over string-typed input gets a direct unit test including its error branches; every mathematical identity the code relies on gets a fast invariant test; every user-facing feature gets one fast API happy-path test plus a slot in the R grid *where an oracle exists* — and an invariant test where it doesn't (fweights being the canonical example).

Concrete missing tests, in priority order: unit tests for vcov-input parsing (dict/str/list/`^` interaction, exhaustive error branches — would have caught two confirmed bugs); a multi-estimation-equals-separate-fits test (demean cache correctness); a `lean=True` contract test; a closed-form 2SLS check without fixed effects (fast, no R); `wald_test` against the F closed form for custom restrictions (after fixing bug B6); a `copy_data=False` mutation-contract test; direct tests for `fit_ols`/`fit_iv` and `vcov_hetero` HC2/HC3 leverage math on tiny constructed matrices.

---

## Part 2 — A development harness for pyfixest

### What the codebase looks like, and the house style it implies

The repository is mid-migration from an organically grown design to a layered one, and the recent code shows clearly what the target style is. The layers: `estimation/api/` (thin user-facing functions, extensive numpy-style docstrings) → `config.py` (one frozen dataclass recording every option) → `plan_.py` (pure planning: formula parsing, model registry, spec expansion) → `runner.py` (orchestration and cache lifecycle) → `models/` (stateful model classes) → `internals/` (pure-ish functions on arrays) → `core/` (Rust kernels with `.pyi` stubs, numba fallbacks in `numba/`). The code the maintainer demonstrably likes — `EstimationConfig`, `OlsFit`/`GlmFit`, the `demeaners.py` strategy dataclasses, `fit_glm_irls` — shares a signature: frozen dataclasses for options and results, keyword-only pure functions for numerics, explicit registries instead of if-chains, `Literal` types validated at the API boundary, comments that cite the econometrics literature (Bergé 2018, Stammann 2018, ppmlhdfe), and differential tests instead of golden values. The legacy patterns being retired are equally identifiable: the 2,500-line `Feols` god class, attributes "enriched outside the class", strings dispatched deep in the stack, `assert` for validation, in-place mutation that changes an attribute's meaning.

That contrast *is* the style guide. Written as rules a contributor or agent can follow:

1. New numerics are pure functions in `internals/` (or Rust in `crates/`) that take arrays and return a frozen dataclass. Model classes orchestrate and store; they do not compute.
2. Options travel as typed values (`Literal` aliases in `internals/literals.py`), validated once at the API boundary with `TypeError`/`ValueError` — never `assert`.
3. An attribute's meaning never changes after it is set. If a transformation produces something semantically new, it gets a new name (`_X` vs `_X_wls`; `_weights` vs `_irls_weights`). Document domains in the class docstring.
4. Every branch over user input ends in `else: raise`. Two of the review's confirmed bugs were fall-through branches.
5. Match `fixest` semantics by default; any deliberate divergence is documented in the docstring with a link (the `ssc` fixef.K note is the model).
6. Estimator code cites its source: paper, equation, or the fixest/ppmlhdfe implementation it mirrors.
7. Tests accompany the change at the right layer: unit test for internals, invariant test for identities, R-grid entry for user-facing behavior with an oracle.

### Guiding contributors — and agents

The single highest-leverage artifact is an `AGENTS.md` at the repo root (with `CLAUDE.md` as a one-line pointer to it, so every coding agent picks it up). Humans get the same content via a "House rules" section in CONTRIBUTING. It should be short enough to fit in an agent's working context — one screen, not a manifesto. A draft is appended below; the essential content is the layer map, the seven rules, the test-tier table with exact commands, and a definition of done.

Two structural aids make the rules self-enforcing rather than aspirational. First, a PR template whose checklist mirrors the definition of done (ruff clean, smoke tier green locally, new-code test policy satisfied, no public-API change without an issue). Second, keep the pixi task descriptions authoritative — `pixi task list` already self-documents; the proposed new tiers below should carry descriptions that say *when* to run them, not just what they do.

### Test tiers: making the fast path known and the slow path scheduled

The pieces already exist (markers, pixi tasks, CI matrix); what's missing is a *smoke* tier at the bottom and a *frequency* split at the top. Proposed ladder:

| Tier | Command | Contents | Budget | When |
| --- | --- | --- | --- | --- |
| 0 smoke | `pixi run test-smoke` (new) | `-m smoke`: invariant tests, parsing/planning units, one happy-path per public entry point, error contracts | < 30 s serial | every edit; agent inner loop; pre-commit |
| 1 quick | `pixi run test-py` (exists) | all Python-only tests minus extended/plots/hac | < 5 min with xdist | pre-push; every PR |
| 2 extended | `test-py-extended`, `plots` (exist) | bootstraps, RI loops, plot rendering | tens of min | PR label / nightly |
| 3 R quick | `pixi run test-r-quick` (new) | curated `test_vs_fixest` slice: full formula list × one base config, plus each dimension varied once against base | < 10 min | every PR touching estimation |
| 4 R full | `test-r-core`/`-fixest`/`-hac`/`-extended` (exist) | the full Cartesian grids | hours-scale | nightly cron + release |

Getting there is mostly bookkeeping: add a `smoke` marker (the 2-second, 162-test set above is the seed — tag `test_weights_wiring`, `test_solvers`, `test_multicollinearity`, `test_exceptions`, `test_plan`, `test_formula_parse`, plus a new `test_smoke_api.py` with one fit-and-tidy per entry point); demote the slowest quick-tier members (`ritest` slow path, decomposition bootstrap variants) to `extended` after measuring with `--durations=25`; and implement the R-quick slice as a `--quick` collection switch or an `r_quick` marker inside `test_vs_fixest.py` so the grid definition stays single-sourced. Two guardrails keep the ladder honest over time: run CI with `--durations=20` so runtime creep is visible in every log, and adopt a soft budget rule — no single test file in tier 1 above ~60 seconds serial, or it moves up a tier. Finally, hook tier 0 into the `prek` pre-commit config alongside ruff, so "the basic tests" are not a convention but a default.

### Small consolidations worth doing alongside

Create `tests/utils.py` with `check_absolute_diff`/`check_relative_diff` (one canonical version), the row-expansion helper, and shared data factories; the three existing copies drift already. Consider `pytest-timeout` with a generous per-test ceiling in tiers 0–1 to convert hangs into failures. And once the tiers exist, update `docs/contributing.qmd`'s test section into the table above — the current text lists commands but not the intent behind each.

---

## Appendix — draft `AGENTS.md`

```markdown
# Working on pyfixest

pyfixest is a Python/Rust implementation of fixest-style fixed-effects
estimation. Match R fixest semantics by default; document deviations.

## Architecture (read top-down)
api/ (user functions) -> config.py (frozen options record) -> plan_.py
(pure planning, MODEL_REGISTRY) -> runner.py (orchestration, caches) ->
models/ (stateful model classes) -> internals/ (pure functions) ->
core/ (Rust kernels; numba fallbacks in estimation/numba/).
`estimation/deprecated/` and root-level `estimation/*_.py` shims are
frozen legacy - do not extend them.

## House rules
1. Numerics = pure functions in internals/ or Rust; arrays in, frozen
   dataclass out. Model classes orchestrate, they do not compute.
2. Validate options at the API boundary (Literal types in
   internals/literals.py); raise TypeError/ValueError, never assert.
3. Never re-bind an attribute to a different meaning. Array domains are
   documented in the Feols class docstring ("Array domains") - read it
   before touching _X/_Y/_weights and friends.
4. Every branch over user input ends in `else: raise`.
5. Public API (function signatures in estimation/api/) is frozen without
   prior discussion in an issue.
6. Cite the source (paper / fixest / ppmlhdfe) for estimator logic.
7. Conventional commits: `fix(scope): ...`, `refactor: ...`, `docs: ...`.

## Tests
- Inner loop:            pixi run test-smoke        (< 30 s, run constantly)
- Before pushing:        pixi run test-py           (minutes, xdist)
- Estimation changes:    pixi run test-r-quick      (needs R env)
- Full R grids run nightly; do not run test-r-core locally unless asked.

New code ships with tests at the right layer: unit test for any internal
function branching on string input (incl. error branches); a fast
invariant test for mathematical identities (see tests/test_weights_wiring.py
and tests/test_ses.py for the pattern); an entry in the R grid for
user-facing behavior that fixest can oracle.

## Definition of done
ruff check + format clean (pixi run lint), test-smoke and test-py green,
tests added per the policy above, docstrings numpy-style, no new public
API without an issue reference.
```
