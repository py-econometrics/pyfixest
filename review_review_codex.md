# Review of GLM and Fable Reviews

## Findings

1. **P0: prediction unseen-category detection is broader than the binned `i()` case.**

   Fable adds an important finding that should be folded into the PR plan, not treated as a duplicate. I verified the `base` decoy-column example: `C(f, contr.treatment(base='a'))` predicts all `NaN` when `newdata` contains a column named `base`. This points at the same fragile parsing area in `pyfixest/estimation/post_estimation/prediction.py`, but it is a separate failure mode from the binned `i()` prediction bug caught by GLM and Codex.

2. **P0/P1: `fixef()` has a public API regression.**

   Fable is right here. I verified that `fixef()` now returns internal wrapped names such as `__fixed_effect__(fe)` and numeric encoded levels instead of user-facing fixed-effect names and labels. For string fixed effects, this is a serious user-visible regression. The relevant code path is in `pyfixest/estimation/models/feols_.py`.

3. **P0: GLM's `i(..., bin=..., bin2=...)` state-collision finding remains valid.**

   GLM's top blocker matches Codex's earlier review: both factor sides share one Formulaic encoder state, and `_apply_binning()` stores one `"bin_mapping"` key. That can make `ref2` or second-variable levels disappear or resolve against the first variable's bins.

4. **P1: fixed-effect dtype mismatch is real, but Fable overstates the symptom.**

   I reproduced this as a low-level `FactorEvaluationError` from pandas merge, not silent `NaN` in the case tested. The recommendation is still good: restore pyfixest-level validation and error messaging around fixed-effect dtype mismatches.

5. **Dependency-risk review: GLM is more balanced; Fable adds one key detail.**

   GLM's Formulaic internals checklist is the stronger dependency-risk section overall. Fable's best addition is the native Formulaic `i()` collision risk, especially where pyfixest may call Formulaic without injecting pyfixest's `i` context. The local Formulaic checkout already has a native `i()` transform.

## Priority Adjustments

I would revise the PR refinement plan priority order to:

1. Rework prediction category detection structurally, not by regex/string scanning. This should fix both binned `i()` prediction and Fable's `base` false-positive bug.
2. Isolate `i()` binning state per variable/side so `bin` and `bin2` cannot overwrite each other.
3. Restore `fixef()`'s public output contract: original fixed-effect variable names, original level labels, and an explicit/tested convention for multi-fixed-effect normalization.
4. Add Formulaic guardrails: compatibility smoke tests, native `i()` collision coverage, and either a conservative upper bound or CI against current Formulaic.
5. Clean up lower-priority issues: unmatched parenthesis `IndexError`, `did2s` parser consistency, contrast-state key helper, copy-paste error messages, and removed or tautological tests.

## Review of the Reviews

GLM is the better dependency-risk review and caught the same two core blockers Codex found. I would downgrade its `did2s` concern from P1 to consistency/polish unless a failing case is added, because the current constructed formulas appear simple.

Fable is the better API-regression review. Its `fixef()` and `base`-column findings are both high-value and verified. I would correct its dtype-mismatch wording before copying it into the plan, but keep the underlying recommendation.
