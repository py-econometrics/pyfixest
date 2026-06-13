# PyFixest Refactoring — Further Opportunities

This is a **companion** to `Refactoring Plan.md` (steps 0–7). That plan already
covers your five in-flight refactors and more:

| Your stated refactor | Covered by |
|---|---|
| Move `Feols.vcov` math into a module | Step 3 (functional core) |
| Unify Poisson / GLM logic | Step 3.2 (`fit_iwls`, fold `Fepois` into `Feglm` family) |
| Clarify the WLS weighting schema | Step 3a / 3b (data-stage naming + weights out of the design matrix) |
| Move post-estimation (wild bootstrap, Gelbach) out | Step 4.1 |
| Execution plan for `FixestMulti` | Step 5 (`ModelSpec` / `expand_specs` / registry / `fit_one`) |

Everything below is **out of scope of that plan** — areas it doesn't touch, plus a
few latent issues found while scanning. Same ground rules apply: no user-facing
API change, golden tests + `test_vs_fixest` green after every PR, small reviewable
diffs. Items are ordered roughly by *value ÷ effort*; the first four are cheap and
high-signal.

---

## A. Quick wins / cruft removal (do first, ~1 PR total)

These are independent, near-zero-risk, and make the codebase visibly tidier for
the next reader.

1. **Delete `pyfixest/debug.py`.** It is a developer scratch file
   (`import pdb; pdb.set_trace()`, a hand-written marginaleffects repro) committed
   at the package root and **shipped to PyPI**. Nothing imports it (verified). If
   it's a useful repro, move it to a gitignored `scratch/` or a `tests/` regression
   case; otherwise delete.

2. **Fix the dead-code capability branch in `Feols.__init__`**
   (`models/feols_.py:319–324`):
   ```python
   if self._weights_name is not None:
       self._supports_wildboottest = False   # ← immediately overwritten on next line
   self._supports_wildboottest = True
   self._supports_cluster_causal_variance = True
   if self._has_weights or self._is_iv:
       self._supports_wildboottest = False
   ```
   The first `if` is dead (overwritten unconditionally). Behavior is correct only
   because the later `if self._has_weights or ...` re-applies it. Delete the dead
   branch. (See item E for the broader cleanup of these flags.)

3. **Rename `utils/_exceptions.py`.** It contains no exceptions — only
   `find_stack_level()`, a warnings helper. Every actual exception lives in
   `errors/__init__.py`. Rename to `utils/_warnings.py` (or fold the one function
   into `utils/utils.py`) so "where do exceptions live" has one answer: `errors/`.

4. **Prune stale bytecode dirs.** `estimation/contracts/`, `estimation/compat/`,
   and `estimation/formula/transforms/` exist on `master` as `__pycache__`-only
   directories (source files removed, `.pyc` left behind from other branches). Add
   `__pycache__/` hygiene or `git clean` these so they don't masquerade as real
   packages. (The `contracts/model_output` stub is relevant to item B below.)

**Effort: small. Risk: none.**

---

## B. An explicit "fittable model" contract (Protocol + result type)

**The single biggest comprehension win not in the existing plan.** Today the fit
pipeline is an *implicit, duck-typed* contract: the `_estimate_all_models` loop
calls `prepare_model_matrix → get_fit → vcov → get_inference → get_performance →
_clear_attributes` on a 7-way `Union[Feols | Feiv | Fepois | Fegaussian | Felogit
| Feprobit | Quantreg | QuantregMulti]`. Nothing declares this interface. The
clearest symptom: **`QuantregMulti` is not a `Feols` subclass but hand-reimplements
all of `prepare_model_matrix`, `to_array`, `drop_multicol_vars`, `wls_transform`,
`demean`, `get_fit`, `vcov`, `get_inference`, `get_performance`,
`_clear_attributes`** just to satisfy the loop. There is no compiler check that it
stays in sync, and a new contributor (or LLM) can't find "the model interface"
because it doesn't exist as a symbol.

You already started this — the empty `estimation/contracts/` dir has leftover
`model_output.py` bytecode. Finish it:

1. Define `FittableModel` as a `typing.Protocol` (in `estimation/contracts/`)
   listing exactly the pipeline methods + attributes the runner depends on. Type
   `fit_one` / `_estimate_all_models` against it instead of the giant Union.
2. Define a frozen `ModelOutput` (or `FitResult`) dataclass for what a fitted
   model exposes downstream (`beta`, `vcov`, `se`, `coefnames`, `N`, `k`, …). This
   is the typed contract your golden/API-snapshot tests (Step 0) assert against,
   and the natural return type of Step 6's `from_arrays`.
3. Make `QuantregMulti` satisfy the Protocol by **composition** (hold a list of
   `Quantreg` models behind the standard results container) rather than
   duck-typing the pipeline.

This rides directly alongside Step 5's registry and Step 6's array core — it's the
*type-level* spine those steps assume but never name. **Effort: medium. Risk: low**
(Protocols are additive; nothing breaks until you tighten the Union).

---

## C. Stop polluting `Feols` with DiD state and monkey-patching

The DiD subsystem (`did/`, a separate `DID(ABC)` hierarchy: `TWFE`, `DID2S`,
`LPDID`, `SaturatedEventStudy`) reaches *into* the core OLS class:

- **`Feols.__init__` carries DiD-only state** it otherwise knows nothing about:
  `_res_cohort_eventtime_dict`, `_yname`, `_gname`, `_tname`, `_idname`, `_att`,
  plus three `_not_implemented_did` placeholder methods (`test_treatment_heterogeneity`,
  `aggregate`, `iplot_aggregate`). A reader of `Feols` meets six attributes and
  three stub methods that have nothing to do with OLS.
- **`did/estimation.py:159–172` monkey-patches a live `Feols` instance**, rebinding
  methods via the descriptor protocol:
  ```python
  fit.aggregate = saturated.aggregate.__get__(fit, type(fit))
  fit.test_treatment_heterogeneity = saturated.test_treatment_heterogeneity.__get__(fit, type(fit))
  fit.iplot_aggregate = saturated.iplot_aggregate.__get__(fit, type(fit))
  ```
  This is invisible to static analysis and to LLM readers — the methods that exist
  on the returned object depend on which `did` function produced it.

Suggestion: make DiD an **extension that composes the result**, not a patch of the
base class. Options, in order of preference:

1. A `DiDResult` wrapper (composition) that *holds* a fitted `Feols` and adds the
   event-study methods — `event_study()` returns that, `feols()` never does. No
   DiD state on `Feols` at all.
2. If returning a `Feols` subclass is required for back-compat, give DiD its own
   `Feols` subclass with those methods defined normally (no `__get__` rebinding),
   and move the six `_*name`/`_att` attributes onto it.

Either way `Feols.__init__` loses its DiD block. The existing plan's tree (Step 7)
doesn't mention `did/` at all — this is genuinely uncovered. **Effort: medium.
Risk: medium** (the monkey-patch semantics need DiD tests pinned first).

---

## D. Collapse the circular-import workarounds

There are **~14 deferred `import_module(...)` calls inside function bodies**, all of
them symptoms of import cycles the layering doesn't yet prevent:

- `report` ↔ `estimation`: `_result_accessor_mixin._bind_report_methods` and
  `FixestMulti._bind...` import `pyfixest.report` at call time and bind
  `summary/coefplot/iplot/etable` onto **every fitted instance** via
  `functools.partial`. That's a per-object closure purely to dodge a cycle.
- post-estimation ↔ top-level package: `feols_.py`, `fepois_.py`, `feiv_.py`, and
  `post_estimation/ritest.py` do `import_module("pyfixest.estimation")` to
  *re-instantiate models* (ritest/ccv refit resampled data by going back up
  through the public namespace).

The existing plan's Step 7 adds an import-linter to *detect* layering violations,
but doesn't prescribe the fix. Two concrete fixes that remove the cycles:

1. **Invert the report dependency.** `summary/etable/coefplot/iplot` are already
   module-level functions taking `models=[...]`. Make those the canonical path and
   drop the per-instance bound copies (or replace `_bind_report_methods` with a
   tiny mixin method `def summary(self, **kw): from pyfixest.report import summary;
   return summary([self], **kw)` — one lazy import in one place, no per-object
   partials). This also shrinks every fitted object.
2. **Re-fit through the functional core, not the package.** Once Step 6 exists,
   ritest/ccv call `fit_ols(...)` / `from_arrays(...)` directly instead of
   `import_module("pyfixest.estimation")`. The cycle disappears because
   post-estimation depends *downward* on the core, not upward on the API.

Doing D makes Step 7's import-linter contract actually *passable* rather than
aspirational. **Effort: medium. Risk: low.**

---

## E. One capabilities model instead of scattered support flags

`Feols.__init__` sets five per-instance booleans —`_support_crv3_inference`,
`_support_hac_inference`, `_supports_wildboottest`, `_supports_cluster_causal_variance`,
`_support_decomposition` — checked ad hoc deep in `vcov()` and the post-estimation
methods (e.g. the `CRV3` branch raises off `_support_crv3_inference`). They're
really **class-level facts** ("IV doesn't support wildboottest"), not instance
state, and the naming is inconsistent (`_support_` vs `_supports_`).

Replace with a single class-level frozen `Capabilities` dataclass (or a class
attribute set) per model type, queried in one place. Subclasses (`Feiv`, the
`Feglm` family) override declaratively. This removes item A.2's bug class entirely,
makes "what can this model do" a one-line read, and gives post-estimation a single
guard helper instead of scattered `if not self._support_x: raise`. **Effort: small.
Risk: low.** Pairs naturally with item B (capabilities can live on the Protocol).

---

## F. Treat `report/` as the other half of the surface

The existing plan touches `report/` only to move docstrings (Step 4.3) and delete
deprecated relabel helpers (Step 1.7). But it's the entire presentation layer and
it's large: `report/visualize.py` (886 lines), `report/visualize_decomposition.py`
(730), `report/summarize.py` (463), `report/utils.py` (293). These mix three
concerns: (a) assembling the data/table to display, (b) formatting (rounding,
significance stars, labels), and (c) rendering to a specific backend
(`lets_plot` vs `matplotlib`; LaTeX vs GT vs markdown for `etable`).

Suggested pass (after the estimation refactor settles, lower priority):

1. Separate "compute a displayable table object" (pure, testable, backend-agnostic)
   from "render it in backend X". `etable`'s LaTeX/GT/markdown variants and the two
   `visualize` plotting backends are the obvious seams.
2. Factor the shared coef-selection / relabel / rounding logic (currently spread
   across `report/utils.py` and duplicated in `_result_accessor_mixin.tidy`) into
   one formatting module.

This is where most *contributor-facing* friction actually is (people add table/plot
features more often than estimators), so it's worth a named workstream even though
it's not on the critical path. **Effort: medium-large. Risk: low** (output is
golden-test-pinnable).

---

## G. Mirror the package tree in `tests/`

`tests/` is 50 flat `test_*.py` files at the root. Step 0 adds golden + API-snapshot
suites but doesn't reorganize. As the package modularizes (Step 7 renames modules),
let tests follow: `tests/estimation/`, `tests/did/`, `tests/report/`,
`tests/core/`, with the slow R-comparison suite (`test_vs_fixest`,
`r_test_comparisons.R`) marked (`@pytest.mark.against_r`) so contributors can run
the fast suite locally. Purely organizational, but it's the difference between a
contributor finding where to add a test in 10 seconds vs grepping 50 files.
**Effort: small (mechanical). Risk: none.**

---

## Suggested sequencing relative to the existing plan

| When | Items | Rationale |
|---|---|---|
| Anytime / now | **A** (cruft, dead branch, rename) | independent, zero-risk, immediate tidiness |
| With Step 0 | **B** (Protocol + `ModelOutput`) | it's the typed contract Step 0's snapshot tests assert and Steps 5–6 assume |
| With Step 4 | **E** (capabilities), start **D** (report binding) | both are "slim the model class" work |
| With Step 5/6 | finish **D** (re-fit via core), fold **C** (DiD) into the registry/result model | DiD becomes "just another result type"; cycles resolve downward |
| With Step 7 | **G** (test layout) | follows the module renames |
| Separate track | **F** (`report/`) | parallel workstream, off the estimation critical path |

### One-line rationale for each, for the changelog / PR descriptions
- **A** — remove shipped scratch file + dead code; one home for exceptions.
- **B** — name the model interface so it's checkable, not duck-typed.
- **C** — DiD stops leaking into the core OLS class and monkey-patching results.
- **D** — break the report/estimation import cycles the layering can't yet enforce.
- **E** — model capabilities become declarative class facts, not scattered flags.
- **F** — split table/plot *data* from *rendering* in the presentation layer.
- **G** — tests mirror the package so contributors know where things go.

---

# Post-plan: residual smells in `Feols` / `FixestMulti`

The items above (A–G) are areas the existing plan doesn't touch. This section is
different: it reviews the **target state of the existing plan itself** — i.e. the
code *after* `refactor-step3-functional-model` is merged — and lists what still
needs attention in the `feols` → `FixestMulti` → `Feols` path.

Headline finding: the branch successfully slimmed line counts (`Feols` 2652 → 1821,
`FixestMulti` 740 → 630) and extracted the leaf math (`internals/vcov_.py`,
`internals/fit_.py`, `post_estimation/*`), but the **structural** smells that most
hurt readability largely survived. In particular, the plan's own **Step 3a** (a
`FitState` dataclass to end the staged-mutation lifecycle) did **not** make it into
the constructor — `Feols.__init__` on the branch is nearly identical to master.

Line references below are to the **branch** versions of the files.

## Tier 1 — highest impact

### R1. Collapse the multi-phase mutation lifecycle in `Feols`

`Feols.__init__` is still a ~155-line god constructor that initializes **~60
attributes to empty placeholders, grouped by the pipeline stage that later fills
them** — the comments literally read `# set in get_fit()`, `# set in vcov()`,
`# set in get_inference()`, `# set in fixef()`, `# set in get_performance()`. The
object is *born invalid* and becomes valid across five subsequent mutation passes;
a reader holding a `Feols` cannot tell which attributes are live without tracing the
pipeline. This is the single highest-value remaining change.

- Hold the staged outputs as sub-objects instead of splatting them onto `self`:
  `self._fit: OlsFit | None`, `self._inference: Inference | None`,
  `self._performance: Performance | None`. The functional core from Step 3 already
  *returns* these — they're just being unpacked onto `self`. `__init__` then sets
  three `None`s, not sixty arrays.
- `_clear_attributes` (`models/feols_.py:1026`) is today a hand-maintained list of
  ~24 attribute-name strings to `delattr` — it silently leaks memory the day
  someone adds a fit attribute and forgets the list. With sub-state objects it
  becomes `self._arrays = None`: one line, can't drift.
- Retire the orphan: `add_fixest_multi_context` (`models/feols_.py:970`) **has no
  callers on the branch** — dead code whose only purpose was the "enrich the object
  from outside the class" anti-pattern (the `# attributes that have to be enriched
  outside of the class - not really optimal` comment in `__init__` flags the same
  thing). Delete it.

**This is the branch's own Step 3a, finished.** Effort: large but incremental
(one sub-state object per PR). Risk: low, guarded by golden tests.

### R2. Move the `vcov` *orchestration* off the class

The plan extracted the vcov *math* into `vcov_.py` / `vcov_utils.py`, but
`Feols.vcov` (`models/feols_.py:627–867`) is **still a ~240-line dispatcher** with
five near-identical `ssc_kwargs = {...}` blocks (one per vcov type) differing only
in `vcov_type`, `G`, and sign. Make it `compute_vcov(model, vcov_spec) ->
VcovResult` with a table mapping `vcov_type → (G_expr, sign, nesting)`; `Feols.vcov`
shrinks to deparse-input → call → store. Same "extract the orchestration, not just
the leaf math" move already applied to `fit_one`. Effort: medium. Risk: low.

## Tier 2 — the `FixestMulti` ↔ `plan_` ↔ `feols` seam

### R3. Make `fit_one` polymorphic — delete the `isinstance` ladder

`fit_one` (`estimation/plan_.py`) is billed as "one place, one order, for all model
types," but still encodes concrete-type knowledge:
`isinstance(fit, (Felogit, Feprobit, Fegaussian))` → `_check_dependent_variable`,
`isinstance(fit, Feols)` → `wald_test`, `isinstance(fit, Feiv)` → `first_stage`,
plus the self-flagged `#  a little hacky, but works` QuantregMulti special-case for
the vcov `data` argument. Replace with hooks every model implements (default
no-ops): `model.check_inputs()`, `model.post_fit()` (Feiv runs `first_stage`; Feols
runs wald + performance), and a `model.vcov_data` property (resolves the
QuantregMulti hack). Then `fit_one` has zero `isinstance` and adding a model never
touches the runner. **This is what makes the new registry/runner actually "open for
extension."** Effort: medium. Risk: low. Pairs with item **B** (the Protocol can
declare these hooks).

### R4. An `EstimationOptions` dataclass to kill the 35-arg `expand_specs`

`expand_specs` takes **~35 flat keyword parameters**, and
`FixestMulti._estimate_all_models` builds that call by re-unpacking ~30 `self._*`
attributes — a data clump threaded api → `FixestMulti` → `expand_specs` →
`ModelSpec.model_kwargs`. Collapse the per-model config into one frozen
`EstimationOptions` (or a few: `DemeanOptions`, `IwlsOptions`, `QuantregOptions`)
built once in `api/feols.py` and threaded through. This also lets you finish the
plan's own open item (Step 5.4, "fold `_prepare_estimation` into `expand_specs`"):
`_prepare_estimation` (`FixestMulti_.py:143`) is still a ~40-line imperative "set 17
attrs to defaults then overwrite" block. Effort: medium. Risk: low.

## Tier 3 — cheap, high-clarity cleanups (all confirmed to survive the plan)

These overlap with A.2 / C / D / E above; restated here as concrete edits to the
**post-plan** `Feols`.

- **R5. Capability flags.** Five scattered booleans in `__init__`
  (`_support_crv3_inference`, `_support_hac_inference`, `_supports_wildboottest`,
  `_supports_cluster_causal_variance`, `_support_decomposition`), inconsistently
  named (`_support_` vs `_supports_`), still including the **dead-code branch** —
  `if self._weights_name is not None: self._supports_wildboottest = False` is
  immediately overwritten by `= True` on the next line. → one declarative
  class-level attribute, overridden by subclasses. (= item **E** / **A.2**.)
- **R6. DiD out of the base class.** `Feols.__init__` still carries
  `_res_cohort_eventtime_dict`, `_yname/_gname/_tname/_idname/_att`, and three
  `_not_implemented_did` placeholders that `did/estimation.py` monkey-patches via
  `.__get__`. None belongs in the OLS constructor. (= item **C**.)
- **R7. `_bind_report_methods`** still binds four `functools.partial`s + an
  `import_module` on *every* model instance. → four one-line mixin methods importing
  lazily: smaller objects, no per-instance closures, statically visible. (= item
  **D.1**.)
- **R8. Post-estimation docstrings.** `Feols` is still ~1820 lines, now dominated by
  the *signatures + large docstrings* of `wildboottest/ccv/decompose/fixef/predict/
  ritest` whose bodies already moved to `post_estimation/`. Keeping thin public
  methods is right, but move each canonical docstring to its function (single source
  of truth) and leave a one-line `"""Run a wild cluster bootstrap. See ...."""` on
  the method. Feols drops well under 1000 lines and stops duplicating docs.

## If you only do two

**R1** (sub-state objects / end the staged-mutation lifecycle) and **R3**
(polymorphic `fit_one`). R1 is what makes `Feols` *legible* — a clear
"computed vs not-yet-computed" boundary instead of one you simulate in your head.
R3 is what makes the new `plan_`/runner architecture deliver the extensibility it
was built for. Both are squarely "make implicit, stateful, type-switched behaviour
explicit" — the stated goal of the whole effort.

### Sequencing for R1–R8

| When | Items |
|---|---|
| Right after the branch merges | R5, R6, R7 (small, independent, no dependencies) |
| Next | R1 (the big one; do per sub-state object) → then R2 (vcov orchestration) |
| With the registry/runner settled | R3 (polymorphic hooks), R4 (`EstimationOptions`) |
| Anytime | R8 (docstring relocation, mechanical) |
