# Meta-review: Codex and Fable refinement plans

Reviewer: GLM. Date: 2026-07-02.
Files reviewed: `PR_REFINEMENT_PLAN_CODEX.md`, `PR_REFINEMENT_PLAN_FABLE.md`.

## Codex (`PR_REFINEMENT_PLAN_CODEX.md`, 60 lines)

Concise, 4 findings. All correct but the shallowest of the three reviews:
- **bin/bin2 state collision** — confirmed (reproduced: `f2` stays unbinned).
- **predict(newdata) for binned `i()`** — follows from the bin bug.
- **FE dtype mismatch** — matches Fable #4 and my P1.
- **formulaic guardrails** — matches both other reviews.

Misses the `fixef()` public-API regression, the multi-FE min-norm behavior
change, all error-message bugs, the test tautology, and the consistency
issues. Useful as a quick checklist but not sufficient for merge readiness.

## Fable (`PR_REFINEMENT_PLAN_FABLE.md`, 188 lines)

The strongest of the three reviews — significantly more thorough than mine
and Codex's. It caught **five issues I missed**:

### 1. `fixef()` returns internal `ngroup()` codes, not real level labels
Fable #2, confirmed regression. `fixef()` now returns
`{"__fixed_effect__(f1)": {"0": …, "1": …}}` instead of
`{"f1": {"north": …, "south": …}}`. For string FEs users **cannot map codes
back**. I observed this in my own testing but failed to flag it. Public-API
regression — should be P0.

Verified:
```
col='__fixed_effect__(f1)', levels=['0', '1', '2', '3']
  → ngroup codes, NOT 'north'/'south'/'east'/'west'
```

### 2. Multi-FE `fixef()` is min-norm, not reference-dropped
Fable #3, confirmed behavior change. `ensure_full_rank=False` encodes **all**
levels of **all** FEs (verified: 4+3=7 columns for two FEs), making the lsqr
system rank-deficient. Per-level values differ from master and R. The inline
comment "reference level dropped" contradicts the code.

### 3. Five concrete error-message bugs
Fable #5, all verified:
- `parse.py:54` prints `len(self._right_hand_side)` (term count = 3) where it
  means `len(self._formula.rhs)` (part count = 4).
- `parse.py:198` `instruments` raises "Endogenous variables are available…"
  (copy-paste from `endogenous`).
- `utils.py:99` bare `FormulaSyntaxError()` (no message).
- Deprecation warnings read "version.Instead" (missing space before "Instead").

### 4. Test tautology
Fable #6, verified — `assert f_implicit.is_fixed_effects is not None` is
always True (bool is never None); should be
`assert not f_implicit.is_fixed_effects`.

### 5. `_model_matrix_one_hot` passes no `context=`
Fable dep item 3, verified at `feols_.py:1511`. Once formulaic ships its
native `i`, this path silently uses formulaic's `i` instead of pyfixest's.
Also affects `C(f1^f2)` (interacted FE) since no custom operator resolver is
applied.

Fable also correctly **upgrades** my P2 #9 (regex over-match in
`_categorical_levels`) to a P0: it reproduces
`C(f, contr.treatment(base='a'))` predicting all-NaN when a column named
`base` exists — a realistic econometrics scenario. I underrated this.

## Where my (GLM) review was stronger than Fable's

I caught two minor issues Fable didn't mention:
- `did2s.py` constructs `Formula` with the default formulaic parser, not
  `_PARSER` (bypasses `^`/MULTISTAGE).
- `_get_position_of_first_parenthesis_pair` raises `IndexError` (not
  `ValueError`) on unmatched `(`.

Both are minor compared to Fable's catches.

## Bottom line

Fable's plan should drive the refinement. Its findings 1–3 are merge
blockers that Codex and I both missed; its findings 5–7 are concrete polish
items that are trivially fixable. Codex's four items are a strict subset of
what Fable and I already cover. The three reviews are complementary only at
the margins — Fable found everything important.

### Cross-review finding matrix

| Finding | Codex | Fable | GLM |
|---|---|---|---|
| bin/bin2 state collision | ✓ #1 | implicit | ✓ P0 |
| predict(newdata) binned i() NaN | ✓ #2 | implicit | ✓ P0 |
| FE dtype mismatch handling | ✓ #3 | ✓ #4 | ✓ P1 |
| formulaic guardrails / version pin | ✓ #4 | ✓ dep | ✓ P1 |
| `fixef()` returns ngroup codes | — | ✓ #2 P0 | **missed** |
| multi-FE fixef min-norm | — | ✓ #3 | **missed** |
| `_categorical_levels` regex over-match | — | ✓ #1 P0 | ✓ P2 (underrated) |
| Error-message bugs (5 issues) | — | ✓ #5 | **missed** |
| Test tautology `is not None` | — | ✓ #6 | **missed** |
| `_model_matrix_one_hot` no context | — | ✓ dep | **missed** |
| Consistency: string literal / missing context / dup helper / `_fml` intercept | — | ✓ #7 | **missed** |
| `_extract_variable_level` test removed | — | ✓ #6 | ✓ P2 |
| `test_vs_fixest` csw0 TODO commented out | — | — | ✓ P2 |
| did2s.py parser inconsistency | — | — | ✓ P1 |
| `_get_position_of_first_parenthesis_pair` IndexError | — | — | ✓ P1 |
| `__contrasts_<var>__` key sync writer↔reader | — | — | ✓ P1 |
| `_preprocess` IV naive `\|` split inside `[]` | — | — | ✓ P2 |
| `ModelMatrix._collect_data` dedup order | — | — | ✓ P2 |
| `_flatten()` comment accuracy | — | — | ✓ P2 |
