# Refinement plan — `formula-parsing-refactor` (rebuild of PR #1222)

Review date: 2026-07-02, updated same day after cross-checking the Codex and GLM
reviews (`PR_REFINEMENT_PLAN_CODEX.md`, `PR_REFINEMENT_PLAN_GLM.md`). Scope:
`git diff master...formula-parsing-refactor` (8 commits, ~20 files, +684/−499).
Verified against formulaic source at `~/Documents/formulaic` (v1.2.1 + 9 dev
commits). Findings marked ⊕ were surfaced by Codex/GLM and independently
reproduced here; one earlier claim of mine is corrected in §5.

## Verdict

The architecture is right and most of the hard parts are correct: deprecated
fixest syntaxes are converted in `_preprocess` (with warnings), NaN fixed-effect
handling matches master (`ngroup()` → NaN, left-merge → NaN for unseen), the
`Term`-vs-string comparisons are sound, the `transform_state` reuse in `fixef()`
keys correctly through formulaic's `format_expr`, and `predict()` is invariant
to the FE normalization change (null-space shifts cancel row-wise). Numerics vs
R fixest pass.

Merge blockers: findings 1–4 (silent wrong results / public-API regression).

---

## Findings (ranked)

### 1. CONFIRMED BUG — false "unseen category" NaNs in `predict(newdata=...)`
`pyfixest/estimation/post_estimation/prediction.py:54`

`_categorical_levels` extracts variable names with
`re.findall(r"[A-Za-z_]\w*", factor_expr)`, so **every identifier** in the
factor expression (function names, keyword args, quoted level strings) is
treated as a candidate column. Reproduced:

```python
df = ...  # has an unrelated column named "base"
fit = pf.feols("y ~ C(f, contr.treatment(base='a'))", data=df)
fit.predict(newdata=df.head(10))   # → all NaN
```

`base` (continuous) is checked against `f`'s categories → every row flagged
unseen → NaN. Column names like `base`, `treatment`, `levels` are common in
econometrics data.

**Fix:** resolve the factor's actual data variables structurally instead of by
regex — e.g. `formulaic.utils.variables.get_expression_variables(factor_expr)`
filtered to data roots, or intersect with
`rhs_spec.variables_by_source["data"]`. Add a regression test with a decoy
column named `base`/`treatment`.

### 2. ⊕ CONFIRMED BUG — `i(a, b, bin=…, bin2=…)` reuses the first variable's bin mapping
`pyfixest/estimation/formula/transforms/factor_interaction.py:264` (`_apply_binning`)

`_apply_binning` caches a single `bin_mapping` key in the **shared**
`encoder_state` of the whole `i()` call. In factor-by-factor interactions the
second `_encode_factor` call passes `bin2`, sees `"bin_mapping"` already set,
and silently applies the *first* variable's mapping. Reproduced:
`Y ~ i(f1, f2, bin={'low': ['a','b']}, bin2={'hi': ['x','y']})` leaves f2
unbinned (coefnames still contain `f2::x`, `f2::y`; no `hi`). Only triggers
when **both** `bin` and `bin2` are given.

**Fix:** namespace the mapping per source variable (e.g. inside each
`__contrasts_<var>__` substate or `__bin_mapping_<var>__`). Test
`i(a, b, bin=…, bin2=…)` with/without `ref`/`ref2` vs R fixest.

### 3. ⊕ CONFIRMED BUG — `predict(newdata=…)` NaNs valid pre-binning levels of binned `i()`
`prediction.py:57-65` (`_categorical_levels`, `__contrasts_` branch)

The unseen-level check compares **raw** newdata values against the
**post-binning** category list. Reproduced: `Y ~ i(f1, bin={'low': ['a','b']})`
→ `predict(newdata=...)` returns NaN for every row with raw level `a`/`b`
(12/20 in the repro), even though materialization would bin them correctly.

**Fix:** apply the stored `bin_mapping` before the unseen check — or better,
unify with finding 1 by moving unseen-level detection into the transform /
materialization path itself, so there is exactly one writer and one notion of
"seen levels". Tests: `i(cat, bin=…)`, `i(cat, x, bin=…)`, `i(cat1, cat2,
bin2=…)` predict round-trips.

### 4. CONFIRMED REGRESSION — `fixef()` returns internal encodings
`pyfixest/estimation/models/feols_.py:1775-1810`,
`pyfixest/estimation/formula/parse.py:213` (`fixed_effects_wrapped`)

Master returned `{"C(f1)": {"<actual level>": value}}`. The branch returns
`{"__fixed_effect__(f1)": {"0.0": value, ...}}` — keys expose the wrapper
transform and **levels are `ngroup()` codes**, not the original FE values. For
string-valued FEs users cannot map codes back. Toy-data tests hide this
(integer FE values coincide with codes); no test asserts keys or level labels.

Related: `feols_.py:1780` passes `ensure_full_rank=False`, so for ≥2 FEs the
D2 system is rank-deficient (verified: 30+30 columns vs master's 30+29) and
lsqr returns the **min-norm** solution — per-level values differ from master
and from R's reference-based convention. `_sumFE`/`predict()` unaffected. The
comment "treatment coding: reference level dropped" contradicts the code.

**Fix:** decode via the stored `encode_fixed_effects` state
(`_state["__fixed_effect_encoding__"]` maps original value tuples → code);
emit fixest-style keys (`f1`, `f1^f2`) and real level labels, or keep master's
`C(f1)` format. Decide reference-based vs min-norm normalization and pin it
with a value-level test vs `fixest::fixef()` (not just sumFE). Fix the comment.

### 5. Degraded FE dtype-mismatch handling in `predict(newdata=...)` (corrected)
Old dtype warning + `tests/test_errors.py::test_predict_dtype_error` deleted.

Empirically verified behavior matrix (float-valued FE fit):
- newdata FE as `int64` / `int32`-fit vs `int64`-newdata: **works, 0 NaN** —
  the deleted test's exact scenario is now handled correctly (better than
  master's warning). GLM's claim that int32-vs-int64 raises is **wrong**.
- newdata FE as `str`: raises formulaic `FactorEvaluationError` wrapping
  pandas' "trying to merge on str and float64 columns" — a **loud but cryptic
  low-level error** (Codex/GLM right; my earlier "silent all-NaN" claim was
  wrong for this case).
- Silent all-NaN remains possible only for merge-compatible dtypes with
  non-matching values (e.g. `"1.0"` vs `"1"` strings).

**Fix:** catch/pre-check FE dtype compatibility in the predict FE branch and
raise/warn with a pyfixest-level message naming the column; warn when *all*
rows fail to match. Reinstate a test for the new behavior.

### 6. Error-message bugs (all confirmed, low severity)
- `parse.py:54` — prints `len(self._right_hand_side)` (term count of part 1)
  where it means `len(self._formula.rhs)` (number of parts).
- `parse.py:198` — `instruments` raises "Endogenous variables are available
  only..." (copy-paste from `endogenous`).
- `formula/utils.py:99` and `:120` — bare `FormulaSyntaxError()` with no
  message.
- `formula/utils.py:107,127` — deprecation messages lack a space before
  "Instead".

### 7. Test hygiene
- `tests/test_formula_parse.py` (`test_explicit_no_fe_with_iv`):
  `assert f_implicit.is_fixed_effects is not None` is a tautology; should be
  `assert not f_implicit.is_fixed_effects`.
- `tests/test_predict_resid_fixef.py`: `test_extract_variable_level` deleted
  but `_extract_variable_level` is still load-bearing in `fixef()` and its
  regex was rewritten in this PR. Restore (incl. nested-bracket case
  `C(x)[T.['ios', 'android']]`).
- `tests/test_vs_fixest.py:1149-1151`: `csw0(X2, f3) + X2` cases commented out
  with a TODO about duplicate handling — fix the dedup or document the skip. ⊕

### 8. Consistency / cleanup (minor)
- `feols_.py:1753` uses the string literal `"second_stage"`; use
  `_ModelMatrixKey.main` like the other call sites.
- `feols_.py:1783` — D2 `get_model_matrix` passes `context=FORMULAIC_TRANSFORMS`
  only; include `{**self._context}` like the second-stage call.
- `feols_.py:334-338` / `447-451` duplicate the `_fixef` formatting chain;
  extract a helper (single place to keep in sync with `parse.py:118`).
- ⊕ Extract the `__contrasts_<var>__` key format into one helper used by both
  writer (`factor_interaction.py`) and reader (`prediction.py`) — GLM's
  suggestion; subsumed anyway if finding 3's "one writer" fix is taken.
- `_fml` now renders as `Y ~ 1 + X1 | f1` (explicit intercept term); cosmetic
  but user-visible in `summary()`/`etable` — strip or changelog.
- Uncached `Formula` properties re-derive per access; harmless today, optional
  `cached_property` later.

### 9. Pre-existing (not introduced by this PR; fix opportunistically)
- ⊕ `formula/utils.py:33-55` `_get_position_of_first_parenthesis_pair`:
  (a) GLM's finding — raises `IndexError` instead of the intended `ValueError`
  when no closing paren follows (reachable via e.g. `Y ~ sw(a, b( | f1)`);
  (b) additionally (found while verifying): the loop increments **before**
  reading, so the first content character is never examined — content starting
  with `(` mis-parses, e.g. `sw((a+b), c)` returns content `"(a+b"` instead of
  `"(a+b), c"`. Restructure the loop (`for position in range(...)`) fixing both.
  Identical code exists on master.
- ⊕ `_preprocess_fixest_instrumental_variable` splits on **all** `|` including
  inside `[...]`/`(...)`; latent today (such formulas are invalid anyway), use
  a depth-aware split like `_str_split_by_sep` when convenient.
- `_preprocess_fixest_multiple_dependents` triggers on `"+" in dependent`, so a
  single depvar like `log(Y1 + Y2) ~ X` is misrouted into `sw()` expansion and
  errors; check `len(_str_split_by_sep(dependent, '+')) > 1` instead. (Master's
  parser was equally naive — parity, but the fix is one line.)
- `Feols._model_matrix_one_hot` (`feols_.py:1511`) calls formulaic with no
  context — see dependency item 3.

---

## Formulaic dependency risks

The in-code `# formulaic internal:` annotations (commit `ccdea8fd`) cover the
touch points: `StructuredFormula.deps`, the `_hat` suffix convention, the
`(Factor.Kind, state_dict)` `encoder_state` tuple shape, `__contrasts_<var>__`
sub-state keys, direct `encode_contrasts(_state=, _spec=)` calls,
`.__wrapped__` unwrapping, and `ModelMatrix._flatten()`.

One correction (per GLM, verified against formulaic source): `Structured`'s
class docstring says underscore-prefixed methods are, "contrary to Python
convention, still considered public" — only `_flatten()`'s **iteration order**
is unstable, which `_collect_data` already tolerates. Soften the
`model_matrix.py:96` annotation accordingly; that touch point is lower-risk
than the rest.

Actions:

1. **Bump the version floor and add a cap.** `pyproject.toml` still says
   `formulaic>=1.1.0`, but the code needs 1.2.x semantics (root-only
   `required_variables`, MULTISTAGE `.deps` layout, FactorValues unwrapping).
   Use `formulaic>=1.2.0,<1.3` (or `<2` plus the smoke test below). Note:
   Codex/GLM's suggestion to keep the 1.1.0 floor and "test against 1.1.0"
   points the wrong way — 1.1.0 cannot work.
2. **Add a formulaic-internals smoke test** exercising: IV parse
   (`.deps` + `_hat` filtering), `i()`/`C()` predict-newdata round-trip
   (`encoder_state` tuple shape + `__contrasts_` keys), and FE encode/decode —
   so a formulaic bump fails loudly in CI.
3. **Upstream collision heads-up:** formulaic master (post-1.2.1, commit
   `00480ec`) adds a *native* `i` transform to `TRANSFORMS`. Verified layering:
   data > context > TRANSFORMS, so paths passing `FORMULAIC_TRANSFORMS` via
   context are safe. The exposure is `Feols._model_matrix_one_hot`
   (`feols_.py:1511`), which passes **no context**: today `i()` models fail
   loudly there; after upstream releases, formulaic's `i` would be used
   silently with different semantics. Pass
   `context=FORMULAIC_TRANSFORMS | {**self._context}` there now (also fixes
   the pre-existing `C(f1^f2)`-as-Python-XOR hazard in that path).
4. ⊕ (GLM P1 #4) `did2s._did2s_vcov` builds `Formula(_formula=formulaic.
   Formula(…))` with the **default** parser — no `_FixedEffectsOperatorResolver`,
   no `FeatureFlags.ALL`. Safe for its current simple formulas; route through
   the shared `_PARSER` for consistency.
5. Benign upstream changes: `bd5f06a` (C() `reduce_rank` kwarg), `286ef1b`
   (`_materializer` injection param).

---

## Suggested commit sequence for the refinement

1. **fix(predict): unify + structuralize unseen-level detection** — findings
   1 + 3 (one mechanism: detect unseen levels where the encoder state is
   written, not by re-parsing expressions); regression tests: decoy `base`
   column, binned `i()` round-trips.
2. **fix(i): per-variable bin-mapping state** — finding 2 + R-parity tests.
3. **fix(fixef): decode FE levels, settle normalization** — finding 4 + value-
   level test vs R.
4. **fix(predict): pyfixest-level FE dtype error/warning** — finding 5 + test.
5. **polish: error messages, deprecation spacing** — finding 6.
6. **test: tautology fix, `_extract_variable_level` test, csw0 TODO** — finding 7.
7. **chore: context/key consistency, `_fixef` helper, `_flatten` comment** —
   findings 8 + dependency correction.
8. **build: formulaic pin `>=1.2.0,<1.3` + internals smoke test + one-hot/did2s
   parser routing** — dependency items 1–4.
9. (optional, pre-existing) **fix(utils): parenthesis-pair helper** — finding 9.

1–4 are the merge blockers; each commit is independently green and reviewable.
