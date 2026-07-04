# Review of the reviews — Codex & GLM on `formula-parsing-refactor`

Date: 2026-07-02. Reviewed documents: `PR_REFINEMENT_PLAN_CODEX.md`,
`PR_REFINEMENT_PLAN_GLM.md`, cross-checked against my own review
(`PR_REFINEMENT_PLAN_FABLE.md`). Every claim below marked "confirmed" was
reproduced empirically (via `pixi run python`) or verified against the
pyfixest/formulaic sources, not taken on trust.

## Verdict

Both reviews land **two real P0 bugs that my original review missed**. GLM's
review is essentially a superset of Codex's — its P0/P1 items 1–3 repeat
Codex's items 1–3 nearly verbatim (down to the phrasing "ref2 can refer to a
level that is never created"), so they clearly share a source or one saw the
other. GLM adds genuine extra findings beyond Codex, but also carries a few
wrong specifics. Neither review looked at what `fixef()` actually returns,
which remains the largest gap covered only by mine. All confirmed findings are
now merged into `PR_REFINEMENT_PLAN_FABLE.md` (marked ⊕).

## What they got right (all reproduced)

- **`i(a, b, bin=…, bin2=…)` bin-mapping collision** (both; GLM's repro claim
  accurate). Confirmed: `_apply_binning` caches one `bin_mapping` in the shared
  encoder state, so `bin2` silently reuses `bin`'s mapping. Repro shows `f2`
  completely unbinned (`f2::x`, `f2::y` still in coefnames, no `hi`). Real
  silent-wrong-model bug; requires both `bin` and `bin2`. **Best catch in
  either review.**
- **Binned `i()` + `predict(newdata)` → false NaN** (both). Confirmed: 12/20
  predictions NaN for rows whose raw levels were binned during training. Same
  root area as my regex finding — the right fix unifies both (detect unseen
  levels where the encoder state is written).
- **GLM: `Structured._flatten()` is not private.** Confirmed against formulaic
  source: the class docstring says underscore methods are "contrary to Python
  convention, still considered public"; only iteration *order* is unstable. My
  annotation and memory overstated this risk — corrected.
- **GLM: `did2s._did2s_vcov` uses the default formulaic parser**, bypassing
  `_FixedEffectsOperatorResolver` and `FeatureFlags.ALL`. Real latent
  inconsistency (currently safe for its simple formulas).
- **GLM: `IndexError` in `_get_position_of_first_parenthesis_pair`** on
  unmatched parens. Real — but pre-existing on master (identical code), which
  GLM didn't note. While verifying it I found an adjacent bug neither review
  saw: the loop increments before reading, so the **first content character is
  never examined** — `sw((a+b), c)` silently mis-parses (content `"(a+b"`).
- **GLM: commented-out `csw0(X2, f3) + X2` cases** with a TODO in
  `test_vs_fixest.py:1149-1151`. Accurate.
- **GLM: `|`-split inside `[...]`** in `_preprocess_fixest_instrumental_variable`
  is depth-unaware. Accurate as a latent P2.

## Where they're wrong or imprecise

- **GLM's dtype example is wrong**: `int32`-fit vs `int64`-newdata does *not*
  raise — it predicts correctly (0 NaN). Float-fit vs int-newdata (the deleted
  `test_predict_dtype_error` scenario) also just works — better than master's
  warning. The real issue is narrower: str-vs-numeric raises a cryptic
  formulaic `FactorEvaluationError` ("trying to merge on str and float64
  columns"). Codex/GLM had the error type right there — and **my original plan
  was wrong too** (I claimed "silent all-NaN" for that case; it is loud but
  cryptic). Corrected in `PR_REFINEMENT_PLAN_FABLE.md` §5. Silent all-NaN
  remains possible only for merge-compatible dtypes with non-matching values
  (e.g. `"1.0"` vs `"1"` strings).
- **Both keep the wrong formulaic floor**: they recommend `>=1.1.0,<2.0` plus
  a "test against minimum 1.1.0" compatibility matrix. The code needs 1.2.x
  semantics (MULTISTAGE `.deps`, root-only `required_variables`) — 1.1.0
  cannot work; the floor must be raised to `>=1.2.0`, not test-matrixed.
- **GLM's "native `i` will collide" is overstated**: formulaic's materializer
  layering is data > context > TRANSFORMS (verified in `base.py:163-166`), so
  pyfixest's `i` wins wherever `FORMULAIC_TRANSFORMS` is passed via context.
  The only real exposure is `Feols._model_matrix_one_hot` (`feols_.py:1511`),
  which passes no context. Codex's softer "keep i explicit in context" was the
  right instinct.
- **GLM under-rated its own finding #9** ("low-impact"): the
  `_categorical_levels` regex over-match makes a plain
  `C(f, contr.treatment(base='a'))` model return **100% NaN predictions**
  whenever newdata has an unrelated column named `base` (reproduced). That is
  a merge blocker, not a P2. Codex missed it entirely.
- **GLM's suggested did2s test is off-target**: an interacted-FE first stage
  (`| id ^ t`) never reaches the `FML1`/`FML2` construction it is meant to
  lock down — FE parts are handled separately in did2s.
- **GLM's `float(level)` fragility note** is moot on the current branch:
  levels are always numeric `ngroup()` codes. It only becomes relevant if
  `fixef()` is fixed to return real level labels (my finding 4) — in which
  case the mapping must change anyway.
- **GLM claims no test covers deprecated IV syntax with FE parts** — but
  `test_multicol_overdetermined_iv` (tests/test_others.py) exercises
  `Y ~ X2 + f1 | f1 | X1 ~ Z1 + Z2` end-to-end.

## What both missed entirely

- The **`fixef()` public-API regression** — keys are now
  `__fixed_effect__(f1)` and levels are `ngroup()` codes instead of real FE
  values (empirically confirmed; the biggest user-facing issue in the PR).
- The **multi-FE normalization change** from `ensure_full_rank=False`
  (rank-deficient D2 → lsqr min-norm values; 30+30 columns vs master's 30+29;
  diverges silently from master and R since only `_sumFE` is tested).
- The **error-message bugs** (`parse.py:54` wrong variable in message,
  `instruments` copy-paste text, bare `FormulaSyntaxError()`s, deprecation
  spacing).
- The **tautological test assertion** in `test_explicit_no_fe_with_iv`
  (`assert f_implicit.is_fixed_effects is not None`).
- The **`_fml` display change** (`Y ~ 1 + X1 | f1`, explicit intercept term).

## Scorecard

| | Codex | GLM | Fable (mine, v1) |
|---|---|---|---|
| Confirmed unique catches | 2 (bin/bin2, binned predict — shared w/ GLM) | + did2s parser, IndexError, `_flatten` correction, csw0 TODO, `|`-split | regex-NaN severity, fixef regression, min-norm change, error msgs, version floor, `i`-layering analysis |
| Factual errors | 0 (but shallow) | 2 (int32 example; floor/matrix direction) + severity misjudgment on #9 + 2 off-target test suggestions | 1 (dtype "silent NaN" claim — corrected) |
| Depth | Findings asserted, no repros | Line-level, partial repro claims | All findings empirically reproduced |

## Bottom line

Codex is short but everything in it is sound. GLM is broader and mostly
verified, with a few wrong specifics and one badly under-rated finding.
Combined with my review, the merge blockers for the PR are: (1) regex
false-NaN in unseen-category detection, (2) `bin`/`bin2` state collision,
(3) binned-`i()` predict NaN, (4) `fixef()` output regression + normalization
decision. All four, plus the corrections above, are consolidated in
`PR_REFINEMENT_PLAN_FABLE.md`.
