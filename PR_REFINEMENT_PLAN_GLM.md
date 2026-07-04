# Formula Parsing Refactor PR — Refinement Plan

Branch: `formula-parsing-refactor` (8 commits since merge-base `27dfbbad`).
Scope: replace pyfixest's hand-rolled formula parser with formulaic's native
MULTIPART (`|` fixed effects) and MULTISTAGE (`[X ~ Z]` IV) syntax; add
stateful `i()` / `__fixed_effect__` / `log` transforms; rewrite `predict()`
and `fixef()` to reuse the stored `ModelSpec`; normalize interacted FE
representation (`:` → `^`); annotate formulaic-internal touch points.

Installed formulaic: `1.2.1` (declared `>=1.1.0`, no upper bound).
Local formulaic checkout (`/Users/afischer/Documents/formulaic`, ahead of
1.2.1) already adds a **native `i()` operator** — a future collision risk.

---

## P0 — Correctness bugs that silently produce wrong results

### 1. `i(a, b, bin=…, bin2=…)` reuses the first variable's bin mapping for the second

`_apply_binning` (`transforms/factor_interaction.py:264`) stashes a single
`bin_mapping` key in the **shared** `encoder_state` dict for the whole `i()`
call. In factor-by-factor interactions the second `_encode_factor` call
(`_encode_i` case iii, line 163) passes `bin2` but `_apply_binning` sees
`"bin_mapping" already in state` and **reuses the first variable's mapping**
instead of building one from `bin2`. `ref2` can then refer to a level that is
never created.

**Reproduced**: `Y ~ i(f1, f2, bin={'low': ['a','b']}, bin2={'hi': ['x','y']})`
leaves `f2` levels as `x, y, z, w` (unbinned) instead of `hi, z, w`.

**Fix**: namespace the bin mapping per source variable, e.g.
`__bin_mapping_<varname>__`, or store it inside each
`__contrasts_<var>__` substate. Add R-fixest comparison tests for
`i(a, b, bin=…, bin2=…)` with and without `ref`/`ref2`.

### 2. `predict(newdata=…)` returns NaN for valid binned `i()` levels

`_rows_with_unseen_categories` (`prediction.py:11`) checks **raw** `newdata`
values against post-binning categories. A raw level like `"a"` (which training
binned to `"low"`) is flagged as unseen → NaN prediction, even though the
binning transform would map it correctly during materialization.

**Fix**: apply the stored bin mapping before the unseen-level check, or move
unseen-level detection into the same transform path that materializes the
model matrix. Add tests for `predict(newdata=…)` with `i(cat, bin=…)`,
`i(cat, x, bin=…)`, and `i(cat1, cat2, bin2=…)`.

---

## P1 — Should land before merge

### 3. Restore clear fixed-effect dtype mismatch handling

The old predict path warned when FE dtypes differed between fit data and
`newdata`. The new `encode_fixed_effects()` path raises a low-level formulaic
`FactorEvaluationError` from `pandas.merge()` for string-vs-float mismatches
(e.g. `int32` fit data vs `int64` newdata, or categoricals that look numeric).

**Fix**: validate FE dtype compatibility before materializing the FE model
matrix; raise/warn with a pyfixest-level message naming the offending column.
Re-add a focused test (the removed `test_predict_dtype_error` covered the
warning path; its replacement should assert the new behavior).

### 4. `did2s.py` builds `Formula` with the default formulaic parser, not `_PARSER`

`did2s._did2s_vcov` constructs `Formula(_formula=formulaic.Formula(…))`
(`did2s.py:320-331`) without passing `_PARSER`. This bypasses the custom
`_FixedEffectsOperatorResolver` (so `^` would be misparsed) and
`FeatureFlags.ALL` (no MULTISTAGE). Currently safe because did2s formulas are
simple (`Y ~ X1 + C(id) + C(t) - 1`), but it is a latent inconsistency.

**Fix**: route through `Formula.parse` (or `formulaic.Formula(…,
_parser=_PARSER)`) so the same parser/feature-flags apply everywhere. Add a
did2s test that uses an interacted-FE first stage (`| id ^ t`) to lock the
behavior.

### 5. `_get_position_of_first_parenthesis_pair` raises `IndexError` on unmatched `(`

`utils.py:33-55`: the `while position < len(string) and depth` loop increments
`position` inside the body and then indexes `string[position]` without a
bounds check. For `sw(a, b` (missing `)`) it raises `IndexError: string index
out of range` instead of the intended `ValueError("Unmatched '(' …")`.

The regex guard `_MULTIPLE_ESTIMATION_PATTERN` requires a closing `\)`, so
this is only reachable via crafted input — but it is a crash, not a clean
error.

**Fix**: check `position < len(string)` after the increment, or restructure
the loop to `for position in range(position_open, len(string))`.

### 6. `__contrasts_<var>__` sub-state keying must stay in sync writer↔reader

`_encode_factor` writes per-variable contrast state under
`__contrasts_<var>__` (`factor_interaction.py:217`), and `_categorical_levels`
reads it back (`prediction.py:61-65`) to detect unseen levels for predict.
If one side changes the key format (e.g. drops the trailing `__`, or uses
`data.name` vs `factor_name`), unseen-level detection silently breaks → rows
get finite-but-wrong predictions instead of NaN.

**Fix**: extract the key format into a single helper
(`_contrasts_state_key(varname) -> str`) used by both sides, or move the
unseen-level check into the transform itself so there is only one writer.

### 7. Formulaic compatibility guardrails

The PR reaches past formulaic's public API in several places (already
annotated with `# formulaic internal:` — good). Concrete touch points and
their risk:

| Touch point | Location | Risk |
|---|---|---|
| `StructuredFormula` / `.deps[0].lhs`/`.rhs` for MULTISTAGE | `parse.py:138-141,189-201` | Undocumented structure; 2.x could restructure → IV detection breaks (loud failure) |
| `<name>_hat` second-stage rename | `parse.py:163-167` (filter in `exogenous`) | `parser.py:360` appends `_hat`; if the suffix changes, endogenous var leaks into exogenous → materialization error (loud) |
| `encoder_state` values as `(Factor.Kind, state_dict)` 2-tuples | `prediction.py:29-34` | Set by `base.py:778`; used in formulaic's own `model_spec.py:348-352` so semi-stable |
| `FactorValues.__wrapped__` unwrapping | `factor_interaction.py:84,111,235` | Matches formulaic's own usage (`contrasts.py:135-136`); `wrapt` contract is stable |
| Direct `encode_contrasts(_state=, _spec=)` call | `factor_interaction.py:226-234` | Part of `@stateful_transform` injection contract; medium risk |
| `Structured._flatten()` for sub-matrix collation | `model_matrix.py:99` | **Not** deprecated (only `SimpleFormula._flatten` is); order is documented unstable but duplicates are value-identical |
| `DefaultOperatorResolver` subclass + `^` replacement | `fixed_effects_encoding.py:26-48` | Public extension point; low risk |
| `transform_state` reuse in `fixef()` | `feols_.py:1776-1787` | Documented state-reuse mechanism; low risk |

**Fix**:
- Pin or bound the formulaic version range (`formulaic>=1.1.0,<2.0`) until the
  internals are either stabilized upstream or pyfixest stops relying on them.
- Add a compatibility test matrix covering the minimum supported (1.1.0), the
  installed (1.2.1), and the adjacent local checkout (which has a native `i()`
  operator that will collide with pyfixest's `i` in `FORMULAIC_TRANSFORMS`).
- Correct the `model_matrix.py:96-98` comment: `Structured._flatten()` is
  public per the `Structured` docstring ("these are still considered public
  methods"); only its **order** is unstable.

---

## P2 — Robustness / follow-up

### 8. `_preprocess_fixest_instrumental_variable` splits `|` inside `[...]`

`utils.py:95` uses `re.split(r"\s*\|\s*", formula)` which splits on all `|`,
including those inside `[X ~ Z | extra]` (the new formulaic IV syntax). A `|`
inside brackets is not valid formulaic syntax today, so this is latent — but
the split would silently mangle the formula if it ever becomes valid.

**Fix**: use a depth-aware split that respects `[]` (the existing
`_str_split_by_sep` already respects `()`; extend or add a bracket-aware
variant).

### 9. `_categorical_levels` regex over-matches in the `categories` branch

`prediction.py:53-56`: `re.findall(r"[A-Za-z_]\w*", factor_expr)` on
`C(f1, Treatment(reference=2))` yields `['C', 'f1', 'Treatment', 'reference']`,
and any of these that happen to be column names produce false-positive
unseen-level checks. A column literally named `C` already breaks formulaic
itself, so this is low-impact, but a targeted extraction (first arg of `C(...)`)
would be safer.

### 10. `ModelMatrix._collect_data` dedup relies on `_flatten()` order

`model_matrix.py:99-103` concatenates all sub-matrices and keeps the first
occurrence of duplicate columns. Currently safe because duplicates (e.g. `X1`
in both `second_stage` and `first_stage`) are value-identical. If a future
formulaic version applies different transforms per sub-spec, duplicates could
diverge and the silent keep-first behavior would mask the discrepancy.

**Fix**: instead of `pd.concat` + dedup, gather columns by name from each
sub-matrix explicitly (the `_collect_columns` method already does this for
metadata; extend it to data).

### 11. Test coverage gaps introduced by this PR

- `test_extract_variable_level` was **removed** (`test_predict_resid_fixef.py`)
  but `_extract_variable_level` is still used in `fixef()` with a nontrivial
  regex. Re-add as a unit test (no R needed).
- `test_predict_dtype_error` was **removed** (`test_errors.py`). Replace with a
  test asserting the new dtype-mismatch behavior (see P1 #3).
- `test_vs_fixest.py:1148-1149` comments out `csw0(X2, f3) + X2` cases with a
  TODO about duplicate handling. Either fix the dedup or document why it is
  skipped.
- No regression test for `fixef()` with **string-typed** FE levels (verified
  working manually; the `float(level)` mapping in `predict()` is fragile for
  non-numeric FE names — add a guard or test).
- No test for the deprecated IV syntax with multiple FE parts
  (`Y ~ X1 | f1 | X2 ~ Z1`).

---

## Verification checklist

```bash
# Core (no R needed)
pixi run -e default pytest tests/test_formula_parse.py tests/test_others.py -q
pixi run -e default pytest tests/test_errors.py -q

# R-dependent (predict/resid/fixef + vs-fixest parity)
pixi run -e py312-r pytest tests/test_predict_resid_fixef.py -q
pixi run -e py312-r pytest tests/test_vs_fixest.py -q

# New tests to add after fixes
# - i(a, b, bin=…, bin2=…) parity vs R fixest
# - predict(newdata=…) with binned i() terms
# - FE dtype mismatch handling
# - did2s with interacted FE (| id ^ t)
# - _extract_variable_level unit test
# - _get_position_of_first_parenthesis_pair with unmatched paren
```

## Dependency notes

- `pyproject.toml` declares `formulaic>=1.1.0` with **no upper bound**. P1 #7
  recommends `>=1.1.0,<2.0` until internal-touch points are removed or
  stabilized.
- The installed `py312-r` environment uses formulaic `1.2.1`.
- The local `/Users/afischer/Documents/formulaic` checkout is ahead of 1.2.1
  and includes a native `i()` transform; future formulaic releases will
  collide with pyfixest's `i` in `FORMULAIC_TRANSFORMS` unless covered by
  compatibility tests.
