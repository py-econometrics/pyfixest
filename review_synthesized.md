# Synthesized Review of the Review Reviews

Source documents:

- `review_review_codex.md`
- `review_review_fable.md`
- `review_review_glm.md`

## Executive Synthesis

The three second-pass reviews converge on a clear refinement path. The PR should not merge until four areas are fixed or explicitly resolved:

1. prediction-time unseen-category detection,
2. `i()` binning state isolation,
3. multi-FE `fixef()` semantics,
4. Formulaic dependency guardrails and transform-context coverage.

Fable's review contributes the strongest prediction-regression finding, especially the `base` decoy-column prediction bug. GLM contributes the broadest dependency-risk inventory and several useful parser/test hygiene issues. Codex's review is shorter but aligns with the core blockers and correctly corrects several severity/wording points.

## Consensus Merge Blockers

### 1. Prediction unseen-category detection is structurally fragile

All reviews now agree that the current prediction path can mark valid rows as unseen and return `NaN`.

Confirmed failure modes:

- binned `i()` terms: training bins raw levels, but prediction checks raw `newdata` values against post-binning categories;
- regular `C(...)` terms: a decoy column named like a contrast argument, such as `base`, can be misread as a model variable and cause all predictions to become `NaN`.

This should be treated as a P0. The fix should avoid regex/string scanning of formatted coefficient names and instead use structured Formulaic metadata or explicit pyfixest metadata captured when encoders are fitted.

Relevant area:

- `pyfixest/estimation/post_estimation/prediction.py`

### 2. `i(..., bin=..., bin2=...)` shares binning state incorrectly

GLM and Codex independently identify the same silent wrong-model bug: both sides of an interaction share one Formulaic encoder state, and `_apply_binning()` stores a single `"bin_mapping"` key.

Impact:

- the second variable can reuse the first variable's bin mapping;
- `ref2` can reference a level that is never created;
- the design matrix can silently encode a different model than requested.

This should remain a P0 because it changes model specification without a clear error.

Relevant area:

- `pyfixest/estimation/formula/transforms/factor_interaction.py`

### 3. Multi-FE `fixef()` semantics need an explicit decision and tests

Fable and GLM agree that multi-FE output likely changed because all levels of all fixed effects are encoded with `ensure_full_rank=False`, producing a rank-deficient system whose per-level estimates are min-norm values. `_sumFE` may remain correct, but individual FE values can differ from previous pyfixest behavior and from fixest/R conventions.

The PR needs one of two outcomes:

- restore the previous reference-normalized output contract; or
- intentionally adopt min-norm values, document this clearly, and add direct tests against expected values.

## Important Corrections to the Reviews

### Fixed-effect dtype mismatch is loud but too cryptic

Several reviews describe the FE dtype issue differently. The best synthesis is:

- string-vs-numeric FE dtype mismatch raises a low-level `FactorEvaluationError` from pandas/Formulaic;
- it is not necessarily silent `NaN`;
- the bug is still real because pyfixest lost its clearer validation/error path.

Action: restore a pyfixest-level error or warning with a targeted message.

### Formulaic minimum version should likely be raised, not merely matrix-tested

The branch appears to rely on Formulaic 1.2-era behavior. A `>=1.1.0` floor is probably too low. The synthesis recommendation is:

- raise the lower bound to at least `>=1.2.0`;
- add smoke tests for the Formulaic APIs pyfixest relies on;
- consider an upper bound if the PR continues to use Formulaic internals or semi-internal state shapes.

Whether the upper bound should be `<1.3` or `<2.0` is a policy choice. `<1.3` is safer for this PR; `<2.0` is less restrictive but should be backed by CI against current Formulaic.

### Native Formulaic `i()` collision is real but scoped

Formulaic now has a native `i()` transform. The collision risk is not uniform:

- where pyfixest passes its transform context, pyfixest's `i()` should win;
- paths that call Formulaic without pyfixest context remain exposed, especially one-hot model-matrix construction.

Action: ensure every Formulaic call path that may parse pyfixest formulas receives the pyfixest transform context, and add a regression test.

### `did2s` parser inconsistency is lower severity

GLM's `did2s` parser concern is valid as a consistency issue, but the current generated formulas are simple enough that this does not look like a merge blocker without a failing example. Keep it in the refinement plan as P2/P3 unless a concrete regression is produced.

### Parenthesis parser `IndexError` is real but peripheral

The unmatched-parenthesis `IndexError` is real and should be cleaned up, but it appears peripheral to the formulaic refactor's main correctness risks. It belongs in the polish/test-hardening bucket unless it affects normal user formulas introduced by this PR.

## Consolidated Refinement Plan

1. Replace prediction-time category extraction with structured metadata.

   Cover both `i(..., bin=...)` and `C(..., base=...)` with tests that fail on the current branch.

2. Separate `i()` encoder state by variable and interaction side.

   Store distinct bin mappings for the first and second variables, then add a regression test for `i(a, b, bin=..., bin2=..., ref2=...)`.

3. Resolve multi-FE `fixef()` semantics.

   Add direct tests for multi-FE output. Decide and document reference-normalized versus min-norm semantics.

4. Restore clear FE dtype mismatch handling.

   Convert the current low-level Formulaic/pandas error into a pyfixest-facing diagnostic with the offending FE variable and dtype mismatch.

5. Add Formulaic compatibility guardrails.

   Raise the lower bound if needed, add smoke tests for required Formulaic APIs, ensure pyfixest transform context is passed consistently, and decide whether an upper bound is necessary for this release.

6. Fix lower-severity parser and messaging issues.

   Include the unmatched-parenthesis error path, incorrect/copy-pasted formula syntax messages, missing warning spacing, duplicate contrast-state key logic, and the tautological test assertion.

7. Restore or replace removed tests.

   Reintroduce coverage for variable-level extraction behavior, deprecated IV syntax where relevant, skipped `csw0` cases if feasible, and any Formulaic behavior that the refactor depends on.

## Final Recommendation

Use Fable's findings as the spine of the refinement work, GLM's dependency and parser checklist as the secondary audit list, and Codex's corrections as severity calibration. The PR's main risk is no longer just formula parsing; it is the combination of silent wrong design matrices, prediction-time false `NaN`s, and unresolved multi-FE fixed-effect semantics.
