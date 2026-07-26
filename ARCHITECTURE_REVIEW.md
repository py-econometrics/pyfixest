# pyfixest — Architecture & Maintainability Review

Reviewed at `0f608eb6` (master). Evaluated against the architecture `AGENTS.md`
states for itself, not a generic ideal.

---

## 1. Executive summary

The architecture is in good shape and clearly improving. The layering that
`AGENTS.md` describes is real: `api/` modules are thin config builders,
`plan_.py`/`runner.py` are a clean planner/executor pair, `internals/fit_.py`,
`internals/vcov_.py` and `internals/demean_.py` are textbook "numbers happen in
functions" modules with frozen result dataclasses, the formula package is well
encapsulated, and the Python↔Rust boundary is the most consistent seam in the
repo (10 kernels, 10 stubs, wrappers and tests for every one). The recent commit
direction — `Make FixestMulti a pure container`, `Refactor wald_test to use
shared _wald_statistic` — is exactly right.

Three structural problems remain, and all three are in the same place: the model
class and the state it carries.

1. **The `lean=True` / `store_data=False` contract is documented but not
   enforced.** `_clear_attributes` deletes attributes by name; nothing on the
   read side knows. Seven of nine common post-estimation methods raise a raw
   `AttributeError` under `lean=True`, which `AGENTS.md` explicitly forbids.
2. **The vcov option set has four spellings across three modules with no single
   registry**, and the two validation functions disagree with the dispatcher.
   Documented inputs (`vcov=["f1","f2"]`, `vcov="NW"`) produce `UnboundLocalError`
   and `KeyError: None`. Validation is `assert`-based, so it vanishes under `-O`.
3. **Nine model constructors re-declare the same ~18 parameters.** Threading one
   estimation option touches 8 files; `QuantregMulti` reaches for
   `inspect.currentframe()` to avoid typing the list a tenth time.

**Highest-leverage single change:** replace the vcov string-branch chain and its
three ad-hoc option lists with one `VcovSpec` registry that owns the name, the
validator, the ssc rule and the `_vcov_<name>` binding (Finding A2). It is
bounded to one subsystem, removes three real crash paths, kills the worst
stringly-typed seam in the codebase, and it is the seam `AGENTS.md` already
tells contributors to extend most often.

---

## 2. Findings register

| ID | Title | Sev | Evidence | Maintainability impact | Effort |
|---|---|---|---|---|---|
| A1 | `lean`/`store_data` strip state with no read-side contract; 7/9 post-estimation methods crash | **high** | `models/feols_.py:926-956` (`_clear_attributes`); `models/feols_.py:1750,1919,2136`; `_result_accessor_mixin.py:306-319,587` | Any refactor that moves an attribute silently changes which methods survive `lean=True`. Only 4 test occurrences guard it. | M |
| A2 | vcov option set duplicated 4× across 3 modules; validation disagrees with dispatch | **high** | `internals/literals.py:4`; `internals/vcov_.py:16-17`; `models/feols_.py:2319-2377`, `:2380-2455`, `:589-718` | Adding a vcov type means editing 3 lists that already disagree; two documented inputs crash with `UnboundLocalError`/`KeyError`. | M |
| A3 | 9 model constructors re-declare ~18 shared parameters | **high** | `feols_.py:251-271`, `feiv_.py:160-205`, `feglm_.py:49-99`, `felogit_.py:16-70`, `feprobit_.py`, `fegaussian_.py`, `fepois_.py:97-145`, `quantreg_.py:63-104`, `QuantregMulti.py:27-66` | One new option = 8 files (measured for `accelerate`); `QuantregMulti.py:53-62` uses frame introspection to work around it. | M |
| B1 | `Feols` still holds numerics and post-estimation orchestration | med | `feols_.py` 2207 lines / 33 methods; `_vcov_crv3_slow:836-868` duplicates `internals/vcov_.py:177-184`; `ccv:1286-1456`; `fixef:1704-1820` | The class `AGENTS.md` says not to grow is still the biggest module; three clear extractions remain. | M |
| B2 | `get_fit` stage contract is OLS-only; `fit_one` special-cases model classes | med | `feols_.py:557-587` vs `feglm_.py:181-252` vs `quantreg_.py`; `plan_.py:318-341` (`isinstance` on `Feglm`, `QuantregMulti`, `Feiv`; "a little hacky" comment at `:333`) | New estimators must be added to `fit_one`'s isinstance chain, not just the registry. | M |
| B3 | `QuantregMulti` duck-types the model interface and mutates sibling privates | med | `QuantregMulti.py:24,80-175` (writes `._beta_hat`, `._u_hat`, `._hessian` on child `Quantreg` objects); `runner.py:102-106` | Not a `Feols` subclass, so every model-shaped code path needs a bespoke branch. | L |
| B4 | `prediction.py` / `savi.py` take the model object, not arrays | med | `post_estimation/prediction.py` (16 distinct `model._*` reads), `savi.py` (12) | Breaks the seam that makes numerics testable against a reference; both are untestable without a full fitted model. | M |
| B5 | Dead code encoding contradictory state contracts | med | `feols_.py:870-924` (`add_fixest_multi_context`, 0 callers), `:2287-2316` (`_feols_input_checks`, 0 callers), `:1458-1471` (unreachable), `:321-322` (dead assignment), `:359` (`self.na_index` write-only, shadows `self._na_index`) | Presents a second, wrong way to initialize model state to the next reader. | S |
| B6 | `_support_*` capability flags are ad-hoc and partly unread | med | `feols_.py:319-327`, `feglm_.py:113-118`, `feiv_.py:208-211`, `fepois_.py:150-151`, `quantreg_.py:115-118`; `_support_iid_inference` and `_support_decomposition` have zero read sites | No base declaration, so a new model class silently inherits `Feols`' permissive defaults for capabilities it does not have. | S |
| C1 | R-comparison tests (the numeric authority) are 54% of the suite and unavailable without R | low | `pytest --collect-only`: 743/1613 in quick suite; `test_vs_fixest.py` is `against_r_core` | CI covers it (`ci-tests.yaml:50-68` runs the full R matrix), so this is local-iteration friction, not a CI gap. | M |

**Not findings — these are healthy:** `internals/fit_.py`, `internals/vcov_.py`,
`internals/demean_.py`, `estimation/formula/`, `plan_.py`/`runner.py`,
`core/`↔`src/` (all 10 kernels registered, stubbed, wrapped and tested).

---

## 3. Deep dives (high-severity findings)

### A1 — `lean` / `store_data` strip state with no read-side contract

`Feols._clear_attributes` (`feols_.py:926-956`) `delattr`s a hard-coded list of
16 attribute names when `lean=True`. Nothing else in the codebase knows that
list. `AGENTS.md` states the rule plainly — "post-estimation code that needs
`self._data` must raise an informative error, not crash" — but the rule is a
convention with no mechanism behind it. Probing a fitted model:

```
lean=True          → predict AttributeError('_X'), fixef AttributeError('_weights'),
                     resid AttributeError('_u_hat'), vcov AttributeError('_data'),
                     ritest AttributeError('_weights'), get_performance AttributeError('_Y')
store_data=False   → fixef/vcov/ritest AttributeError('_data')
```

Seven of nine methods fail with a raw `AttributeError` naming a private
attribute. The maintainability cost is not the crash — it is that the deleted
set and the required set are never compared. Imagine adding a post-estimation
method that needs `_scores` (an SAVI variant, a new HAC-adjusted test). Nothing
tells you `_scores` is on the `lean` deletion list; the method works in every
test you write, and breaks only for users who passed `lean=True`. The same trap
runs backwards: dropping a name from the deletion list to fix one method
silently enlarges every lean model's memory footprint, which is the entire
purpose of the flag.

**Target design.** Make the requirement declarative and check it at the one
place that already knows the model is stripped.

1. Add `_LEAN_CLEARED: frozenset[str]` and `_DATA_CLEARED: frozenset[str]` as
   class constants on `Feols`, and have `_clear_attributes` iterate them (pure
   refactor, no behavior change).
2. Add a small `_require(self, *attrs, feature: str)` helper on `Feols` that
   raises a `LeanModelError` (new class in `pyfixest/errors/__init__.py`) naming
   the method, the missing attribute and the flag that removed it.
3. Call `_require` at the top of the eight methods that need stripped state:
   `predict`, `fixef`, `resid`, `vcov`, `ritest`, `ccv`, `wildboottest`,
   `get_performance`, `decompose`.
4. Add one parametrized test over `(method, lean, store_data)` asserting the
   error type and that the message names the flag — this is exactly the
   "parametrized matrix" style `AGENTS.md` prefers, and it makes the deletion
   list and the requirement list mutually checkable from then on.

Step 1 alone is a safe quick win; steps 2–4 are one bounded PR.

### A2 — The vcov option set has no single source of truth

The set of legal `vcov` values is currently written down four times:

| Where | Contents |
|---|---|
| `internals/literals.py:4` `VcovTypeOptions` | `iid, hetero, HC1, HC2, HC3, nid` — **missing `NW`, `DK`** |
| `internals/vcov_.py:16-17` | `HeteroVcovTypeOptions`, `HacVcovTypeOptions` |
| `models/feols_.py:2357-2369` `_check_vcov_input` | `iid, hetero, HC1, HC2, HC3, NW, DK, nid` (as an `assert`) |
| `models/feols_.py:2416-2441` `_deparse_vcov_input` | the `if/elif` chain that actually dispatches |

`VcovTypeOptions` is the one `AGENTS.md` tells you to edit first, it is the
annotation on all four public API signatures — and it has already drifted out of
date, because it is never enforced at runtime (`_validate_literal_argument` is
never called on `vcov`). Three verified consequences:

```python
fit.vcov(["f1", "f2"])   # accepted by _check_vcov_input → UnboundLocalError in _deparse_vcov_input
fit.vcov("NW")           # no boundary check → KeyError: None from self._data[self._time_id]
python -O; fit.vcov({"BOGUS": "f1"})   # asserts stripped → UnboundLocalError
```

The list branch is dead at the `feols()` boundary (`_estimation_input_checks`
rejects it) but live post-estimation, so `_check_vcov_input` carries validation
for a shape `_deparse_vcov_input` cannot handle. The `NW` case is worse: the
`time_id` check at `feols_.py:2372-2377` only fires when `vcov_kwargs is not
None`, so the most likely user error skips validation entirely and surfaces as
`KeyError: None` from deep inside `_vcov_hac`.

Imagine adding a Conley spatial-HAC estimator. Following the `AGENTS.md` recipe
you edit `literals.py`, add a branch to `_check_vcov_input`, add a branch to
`_deparse_vcov_input`, add an `elif` to the 109-line `vcov()` body with its own
`_make_ssc_kwargs` call, add `_vcov_conley`, and thread it through
`FixestMulti.vcov`. Six edits in three files, three of which are parallel lists
that nothing keeps in sync — and you inherit the same unbound-local failure mode
for any input shape you forget.

**Target design.** One registry, one dispatcher. The `_vcov_<name>` method-override
seam is already good (`Quantreg` overrides `_vcov_iid`, `_vcov_hetero`,
`_vcov_nid`, `_vcov_crv1` cleanly) — keep it and put a table in front of it.

1. In `internals/vcov_utils.py`, add a frozen `VcovSpec` dataclass:
   `detail` (the user-facing name), `family` (`iid`/`hetero`/`HAC`/`CRV`/`nid`),
   `is_clustered`, `method_name` (`"_vcov_hetero"`), `ssc_G` (a callable taking
   the model), and `validate` (a callable raising `ValueError` on bad
   `vcov_kwargs`/model combinations). Build `VCOV_REGISTRY: dict[str, VcovSpec]`.
2. Derive `VcovTypeOptions` from the registry keys so it can never drift again.
3. Rewrite `_check_vcov_input` to look up the registry and call `spec.validate`,
   raising `ValueError` (not `assert`) with the allowed values listed. Move the
   `NW`/`DK` `time_id` requirement into the NW/DK spec's `validate` so it fires
   when `vcov_kwargs is None`. Either implement the list form as sugar for
   `{"CRV1": "a+b"}` or reject it in both functions — do not leave it half-supported.
4. Reduce `_deparse_vcov_input` to a registry lookup plus the `^`→`_` cluster
   rename, and reduce `Feols.vcov`'s `if/elif` chain (`:666-714`) to: look up
   spec → `get_ssc(**self._make_ssc_kwargs(vcov_type=spec.family, G=spec.ssc_G(self)))`
   → `getattr(self, spec.method_name)()`. The CRV branch keeps its
   `prepare_cluster_state`/`run_crv_loop` path.
5. Tests: `test_errors.py` for each malformed input (list, unknown string,
   `NW` without `time_id`, bad dict key), then `pixi run test-r-fixest` and
   `test-r-hac` to prove the numbers are untouched.

### A3 — Nine model constructors re-declare the same parameter list

Measured parameter counts: `Feols` 18, `Feiv` 18, `Feglm` 23, `Felogit` 22,
`Feprobit` 22, `Fegaussian` 22, `Fepois` 22, `Quantreg` 22, `QuantregMulti` 23 —
sharing ~18 base parameters. `felogit_.py` is the clearest case: 72 lines, of
which ~68 are a verbatim re-declaration and forwarding of `Feglm`'s signature.
The only real content is `family=LOGIT` and `self._method = "feglm-logit"`.
`Feprobit` and `Fegaussian` are the same file with two words changed.

The cost is measurable. Threading the `accelerate` option touches eight files
(`api/feglm.py`, `config.py`, `plan_.py`, `internals/fit_glm_.py`, `feglm_.py`,
`felogit_.py`, `feprobit_.py`, `fegaussian_.py`), five of them purely to retype
the parameter. `QuantregMulti.__init__` (`:53-62`) gives up and harvests its own
arguments with `inspect.currentframe()` / `getargvalues` to forward them —
a reflection hack whose only justification is that the list is too long to
repeat again. That hack is also a latent trap: it forwards *everything* except
three named exclusions, so a future `Quantreg`-only parameter added to
`QuantregMulti` will be silently forwarded to `Quantreg` too.

Imagine adding a `nthreads` option that every estimator should honour. Today
that is `config.py` + `plan_._build_model_kwargs` + nine signatures + eight
`super().__init__` call sites, with mypy catching only the ones you typo.

**Target design.** Pass the already-frozen config object instead of re-exploding it.

1. Introduce a frozen `ModelInit` dataclass in `estimation/models/` holding the
   ~18 base fields (`FixestFormula`, `data`, `ssc_dict`, `drop_singletons`,
   `drop_intercept`, `weights`, `weights_type`, `collin_tol`,
   `lookup_demeaned_data`, `lookup_preconditioner`, `solver`, `demeaner`,
   `store_data`, `copy_data`, `lean`, `context`, `sample_split_var`,
   `sample_split_value`).
2. Give `Feols.__init__` the signature `(self, init: ModelInit)`; each subclass
   becomes `(self, init: ModelInit, *, tol, maxiter, ...)` carrying only its own
   extras. `Felogit`/`Feprobit`/`Fegaussian` collapse to ~10 lines each; the
   three could then plausibly become one `Feglm` with a `family` argument, since
   that is all that distinguishes them.
3. `plan_._build_model_kwargs` builds a `ModelInit` plus a small extras dict;
   `MODEL_REGISTRY.needs` keeps doing exactly what it does now.
4. Drop `QuantregMulti`'s frame introspection — it forwards `init` verbatim.
5. This is internal only (`AGENTS.md`: users never construct these classes), so
   no deprecation shim is needed beyond the existing `estimation/feols_.py`
   re-exports. Verify with `test_plan.py`, `test_modular_runner.py`,
   `test_api.py`, then the full R matrix.

Do A3 **after** A1 and A2 — it touches every model file, so it wants a quiet
tree.

---

## 4. Action plan

Each item is independently mergeable. No phase depends on a later one.

### Phase 0 — Quick wins (each < 1 day, low risk, no API change)

**P0-1. Delete dead code (B5).**
*Files:* `models/feols_.py`.
*Steps:* remove `add_fixest_multi_context` (`:870-924`), `_feols_input_checks`
(`:2287-2316`), the unreachable block after `return` in `ccv` (`:1458-1471`),
the dead `self._supports_wildboottest = False` (`:321-322`), and the write-only
`self.na_index` (`:359`) — after confirming `Feglm.prepare_model_matrix:161`
is its only other writer and migrating that line to `self._na_index` or dropping
it. *Risk:* low; all have zero callers (verified by grep over `pyfixest/` and
`tests/`). *Verify:* `pixi run test-py`, then `pixi run test-r-core`.

**P0-2. Sync `VcovTypeOptions` with reality (A2, partial).**
*Files:* `internals/literals.py`.
*Steps:* add `"NW"`, `"DK"` to the Literal. *Risk:* none at runtime (annotation
only); makes the four public signatures honest immediately. *Verify:*
`pixi run -e lint prek run mypy --files pyfixest/estimation/internals/literals.py`.

**P0-3. Close the `NW`/`DK` `time_id` hole (A2, partial).**
*Files:* `models/feols_.py:2372-2377`.
*Steps:* hoist the `time_id` requirement out of the `vcov_kwargs is not None`
guard so `vcov="NW"` with no kwargs raises the existing `ValueError` instead of
`KeyError: None`. *Risk:* low — turns a crash into the intended error. *Verify:*
add the case to `tests/test_errors.py`; `pixi run test-r-hac`.

**P0-4. Convert `_check_vcov_input` asserts to raises (A2, partial).**
*Files:* `models/feols_.py:2319-2377`.
*Steps:* replace the five `assert`s with `ValueError`/`TypeError` listing the
allowed values, per the `AGENTS.md` error convention. *Risk:* low; the messages
are already written. *Verify:* `pixi run -e py312-r pytest tests/test_errors.py -q --no-cov`.

**P0-5. Declare the capability flags on the base class (B6).**
*Files:* `models/feols_.py`, `models/_result_accessor_mixin.py`.
*Steps:* declare all five `_support*` flags as annotated class attributes with
explicit defaults; delete `_support_iid_inference` and `_support_decomposition`
if they still have no read sites, or wire them into `_vcov_iid`/`decompose`.
*Risk:* low. *Verify:* `pixi run test-py`.

### Phase 1 — Local refactors (bounded to one subsystem, test-covered)

**P1-1. The vcov registry (A2).** Steps 1–5 of the A2 deep dive.
*Files:* `internals/vcov_utils.py` (new `VcovSpec` + registry),
`internals/literals.py`, `models/feols_.py` (`vcov`, `_check_vcov_input`,
`_deparse_vcov_input`), `quantreg/quantreg_.py` (register `nid`),
`FixestMulti_.py` (unchanged if the signature holds).
*Risk:* medium — this is the inference path. Mitigated by the fact that the
numerics (`internals/vcov_.py`) are untouched; only selection moves.
*Verify:* `test-r-fixest`, `test-r-hac`, `test-r-core`, `tests/test_ses.py`,
`tests/test_crv1_vcov.py`, `tests/test_errors.py`.

**P1-2. The lean/store_data contract (A1).** Steps 1–4 of the A1 deep dive.
*Files:* `models/feols_.py`, `models/_result_accessor_mixin.py`,
`errors/__init__.py`, `tests/test_api.py` (or a new
`tests/test_model_lifecycle.py`).
*Risk:* low-medium — user-visible exception *type* changes from `AttributeError`
to a pyfixest error on paths that were crashing anyway. Worth one line in the
changelog. *Verify:* new parametrized lifecycle test; `pixi run test-py`.

**P1-3. Finish carving numerics out of `Feols` (B1).**
*Files:* `models/feols_.py`, `internals/vcov_.py`,
`post_estimation/ccv.py`, `post_estimation/fixef.py` (new).
*Steps:* (a) make `_vcov_crv3_slow` call the existing
`internals/vcov_.py::_jackknife_vcov` instead of repeating the loop
(`feols_.py:861-868`); (b) move the CCV aggregation body (`ccv:1380-1456`) into
`post_estimation/ccv.py`, leaving validation and the `self._` unpack in the
method — `ritest` is the template; (c) move the `fixef()` matrix construction
(`:1750-1818`) into a new `post_estimation/fixef.py` taking arrays.
*Risk:* low, mechanical; each sub-step is its own commit.
*Verify:* `tests/test_ccv.py`, `tests/test_predict_resid_fixef.py`,
`tests/test_ses.py`, `test-r-core`.

**P1-4. Give `prediction.py` and `savi.py` array signatures (B4).**
*Files:* `post_estimation/prediction.py`, `post_estimation/savi.py`,
`models/feols_.py`, `models/_result_accessor_mixin.py`.
*Steps:* replace `model=self` with explicit keyword arrays; the methods unpack
`self._` into locals, as `Feols.get_fit` → `fit_ols` already does. Where the
argument count gets unwieldy (prediction reads 16 attributes), introduce a small
frozen `PredictionInputs` dataclass rather than passing the model.
*Risk:* low. *Verify:* `tests/test_predict_resid_fixef.py`, `tests/test_savi.py`,
`tests/test_savi_vs_avlm.py`.

### Phase 2 — Structural changes (only if the payoff exceeds the risk)

**P2-1. `ModelInit` (A3).** Steps 1–5 of the A3 deep dive. Recommended: the
payoff (one new option = 2 files instead of 8, three near-identical GLM
subclasses collapse, one reflection hack dies) clearly exceeds the risk, since
the change is internal and mypy-checked end to end. *Verify:* `test_plan.py`,
`test_modular_runner.py`, `test_api.py`, then the full R matrix.

**P2-2. A model protocol for `fit_one` (B2/B3).** Only worth doing after P2-1.
*Steps:* define a `FittableModel` Protocol (`prepare_model_matrix`, `get_fit`,
`vcov`, `get_inference`, `_clear_attributes`, plus optional hooks
`post_fit_hook`, `check_dependent_variable`) and move `fit_one`'s three
`isinstance` branches (`plan_.py:318-341`) onto the classes: `Feglm` implements
the dependent-variable check, `Feiv` implements `first_stage` as a post-fit
hook, `QuantregMulti` implements the data accessor that the "a little hacky"
comment works around. *Risk:* medium. *Payoff:* adding an estimator becomes a
registry entry plus a class, with no edits to the runner.
*Defer* `QuantregMulti`'s deeper problem (mutating sibling privates) — it is a
genuine algorithmic requirement of the CFM quantile process and needs its own
design pass, not a refactor.

**P2-3. Not recommended: a shared estimator base beyond `ModelInit`.** The
`Feols` → `Feiv`/`Feglm`/`Quantreg` inheritance chain is doing real work
(shared vcov dispatch, shared accessors) and the overrides are well-scoped.
Introducing composition or an abstract estimator interface would be a big-bang
rewrite with no failure it demonstrably prevents.

---

## 5. Do-not-touch list

Deliberate per `AGENTS.md` — do not "fix" these:

- **`self._underscore` model state.** House style, stated explicitly. A1 makes
  the *lifecycle* enforceable; it does not replace the convention with properties.
- **Compat shims** `estimation/feols_.py`, `feiv_.py`, `fepois_.py` and
  `estimation/deprecated/` (still imported by `estimation/__init__.py:19`).
- **The four separate API docstrings.** "there is no shared docstring" is a
  documented choice; do not DRY them into a decorator.
- **`fixest`-mirroring user-facing behavior**: `vcov` string spellings, `ssc`
  semantics, defaults, `_method` names as they appear in output. A2 changes how
  the option set is *stored*, never which strings users may pass.
- **Long solver loops** — the IRLS loop in `internals/fit_glm_.py`, Frisch–Newton
  in `quantreg/frisch_newton_ip.py`, LSMR in `torch/`. Explicitly sanctioned as
  exceptions to "keep functions short"; splitting them hurts `torch.compile`.
- **`estimation/torch/` and `estimation/numba/` duplication** of the demean
  kernels — these are deliberate alternate backends and NumPy/numba references
  for the Rust kernels.
- **The `core/`↔`src/` boundary.** Reviewed and consistent: every kernel in
  `src/lib.rs` is registered, stubbed in `_core_impl.pyi`, re-exported under a
  clean alias, and tested. No change needed.
- **The 54% R-gated test suite (C1).** CI runs the full R matrix on every PR
  (`ci-tests.yaml:50-68`); the split is a deliberate local-speed tradeoff.

---

## 6. Open questions for the maintainer

1. **`vcov` as a list.** `_check_vcov_input` validates `vcov=["f1","f2"]` as
   column names, `_deparse_vcov_input` cannot handle it, and `feols()` rejects it
   at the boundary. Was this intended as an alias for `{"CRV1": "f1+f2"}` — the
   natural reading of a list of cluster columns — or is it vestigial? P1-1 needs
   the answer to either implement it or delete both half-branches.
2. **Exception type change under `lean=True`.** P1-2 converts raw
   `AttributeError` into a pyfixest error. Anyone catching `AttributeError`
   today is relying on an accident, but it is technically a behavior change —
   changelog note, or is a minor-version note enough?
3. **Appetite for `ModelInit` (P2-1).** It rewrites nine constructor signatures
   in one PR. These are non-user-facing per `AGENTS.md`, but it will conflict
   with any in-flight branch that touches a model class. Is there a window?
4. **Collapsing `Felogit`/`Feprobit`/`Fegaussian`.** After P2-1 they are ~10
   lines each and differ only by their `family` constant. Fold them into `Feglm`
   with a family argument, or keep separate classes so `_method` and the
   quartodoc reference entries stay one-per-family?
5. **`_support_iid_inference` and `_support_decomposition`** are set in four
   classes and read nowhere. Were they meant to gate `_vcov_iid` and
   `decompose()` (in which case P0-5 should wire them up, and `Feiv`/`Feglm`
   would start raising where they now silently compute), or are they abandoned?
6. **`QuantregMulti`'s place in the type system.** It is returned by
   `fit_one`, unwrapped in `runner.run_estimation:102-106`, and is not a `Feols`.
   Should it become a planner-level concern (the planner emits N `Quantreg`
   specs sharing a fitted-state cache) rather than a model class? That would
   remove the special cases in both `fit_one` and `run_estimation`, but it is a
   design change, not a refactor.
