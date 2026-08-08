# Formula Parsing Refactor PR Refinement Plan

## Review Findings To Address Before Merge

1. Fix `i(..., bin=..., bin2=...)` state collision.
   - `factor_interaction._apply_binning()` stores one `bin_mapping` in the shared
     encoder state for the whole `i()` call. In factor-by-factor interactions, the
     first variable's `bin` mapping is then reused for the second variable's `bin2`
     mapping, so `ref2` can refer to a level that is never created.
   - Store bin mappings per source variable, e.g. `__bin_mapping_<var>__`, or inside
     each `__contrasts_<var>__` substate.
   - Add regression tests for `i(a, b, bin=..., bin2=...)`, with and without
     `ref`/`ref2`, compared against R fixest.

2. Fix `predict(newdata=...)` for binned `i()` terms.
   - `_rows_with_unseen_categories()` checks raw `newdata` values against
     post-binning categories. Valid raw levels such as `"small"` are flagged as
     unseen when training binned them to `"not_large"`, causing predictions to
     return `NaN`.
   - Apply the stored bin mapping before unseen-level checks, or move unseen-level
     detection into the same transform path that materializes the model matrix.
   - Add tests for `predict(newdata=...)` with `i(cat, bin=...)`,
     `i(cat, x, bin=...)`, and `i(cat1, cat2, bin2=...)`.

3. Restore clear fixed-effect dtype mismatch handling.
   - The old prediction path warned when fixed-effect dtypes differed between fit
     data and `newdata`. The new `encode_fixed_effects()` path can raise a low-level
     formulaic `FactorEvaluationError` from `pandas.merge()` for string-vs-float
     mismatches.
   - Validate FE dtype compatibility before materializing the FE model matrix, and
     raise/warn with a pyfixest-level message that tells users which FE column differs.

4. Add formulaic compatibility guardrails.
   - The PR relies on formulaic internals: multistage `.deps` and `_hat` naming,
     `ModelSpec.encoder_state` tuple shape, `FactorValues.__wrapped__`, direct
     `encode_contrasts(..., _state=..., _spec=...)`, and structured model-matrix
     `_flatten()`.
   - Either narrow the supported formulaic range or add a compatibility test matrix
     covering the minimum supported formulaic version, the currently installed
     version, and the adjacent local formulaic checkout.
   - Keep pyfixest's `i()` transform explicit in context because the local formulaic
     checkout already contains a native `i` transform.

## Suggested Test Checklist

- `pixi run -e py312-r pytest tests/test_i.py`
- `pixi run -e py312-r pytest tests/test_others.py::test_predict_newdata_i_transform tests/test_others.py::test_predict_newdata_unseen_category`
- `pixi run -e py312-r pytest tests/test_predict_resid_fixef.py`
- A new non-R unit test for binned `i()` prediction that asserts valid binned levels
  remain finite.
- A new non-R unit test that `i(a, b, bin=..., bin2=...)` fits and stores separate
  bin mappings for both variables.

## Dependency Notes

- `pyproject.toml` currently declares `formulaic>=1.1.0` with no upper bound.
- The installed py312-r environment uses formulaic 1.2.1.
- The local `/Users/afischer/Documents/formulaic` checkout is ahead of 1.2.1 and
  includes a native `i()` transform, so future formulaic releases may collide with
  pyfixest assumptions unless covered by compatibility tests.
