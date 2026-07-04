# Formulaic Guardrails Review Plan

This plan reviews the current stacked PR commit by commit, while cross-reading
the relevant Formulaic 1.2.1 internals. The goal is to understand whether
pyfixest is relying on Formulaic behavior safely, and where the remaining risk
lives.

## Setup

```bash
git log --oneline --reverse origin/formula-parsing-refactor..HEAD
git show --stat <sha>
git show <sha>
```

The stack currently targets Formulaic 1.2.x. The pinned dependency is in
[`pyproject.toml`](pyproject.toml).

## Formulaic Concepts To Read First

Read these once before reviewing the commits.

### Formula Parsing And MULTISTAGE

- [Formulaic parser feature flags](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/parser/parser.py#L70)
- [`ALL` includes `MULTISTAGE`](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/parser/parser.py#L75)
- [MULTISTAGE parser creates `_hat` terms and `.deps`](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/parser/parser.py#L354-L371)

`MULTISTAGE` is the parser feature behind formulas like `[X2 ~ Z1]`. Formulaic
turns the endogenous variable into an estimated second-stage term named
`X2_hat`, and stores the first-stage formula in a structured dependency under
`.deps`.

### Formula Containers

- [SimpleFormula](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/formula.py#L331)
- [StructuredFormula](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/formula.py#L640)
- [Structured._flatten](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/utils/structured.py#L193)

`SimpleFormula` represents a flat term list. `StructuredFormula` is used for
multi-part formulas and MULTISTAGE structures. `_flatten()` is private; pyfixest
uses it only to collect matrix leaves and does not rely on order.

### Model Matrices And Model Specs

- [ModelMatrix](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_matrix.py#L26)
- [ModelMatrices](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_matrix.py#L77)
- [ModelSpec](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_spec.py#L31)
- [ModelSpec.get_model_matrix](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_spec.py#L496)

`ModelSpec` stores the choices made when building a model matrix. Reusing the
stored spec is the right prediction-time path because it preserves encodings,
contrasts, and transform state.

### Categorical Metadata

- [ModelSpec.factor_variables](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_spec.py#L307)
- [ModelSpec.factor_contrasts](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/model_spec.py#L326)
- [encoder_state write location](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/materializers/base.py#L778)

For formulaic-native categorical terms, prefer `factor_contrasts` and
`factor_variables`. Raw `encoder_state` is still needed for pyfixest's custom
`i()` transform state.

### Stateful Transforms And Contrasts

- [stateful_transform](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/utils/stateful_transforms.py#L21)
- [`_state` / `_spec` injection](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/utils/stateful_transforms.py#L49-L84)
- [encode_contrasts](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/transforms/contrasts.py#L95)
- [ContrastsState](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/transforms/contrasts.py#L929)

Formulaic passes `_state` and `_spec` into stateful transforms. pyfixest's
custom `i()` transform calls `encode_contrasts` directly with those same state
objects to preserve Formulaic-compatible contrast behavior.

### FactorValues Wrapping

- [FactorValues](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/materializers/types/factor_values.py#L93)
- [Formulaic unwrapping nested FactorValues](https://github.com/matthewwardrop/formulaic/blob/v1.2.1/formulaic/materializers/types/factor_values.py#L137)

`FactorValues` is a `wrapt.ObjectProxy` carrying metadata about kind, column
names, span of intercept, and custom encoder callbacks.

## Commit Review

### 1. `eb1e0653` - Restore Fixed-Effect Predict Dtype Guard

Review:

- [`_check_fe_dtype_compatibility`](pyfixest/estimation/models/feols_.py)
- `predict(newdata=...)` call site in [`feols_.py`](pyfixest/estimation/models/feols_.py)

Questions:

- Does the guard reject numeric-vs-non-numeric FE mismatches?
- Does it allow numeric-vs-numeric differences like int vs float?
- Is the error clearer than the downstream Formulaic/materializer failure?

Suggested check:

```bash
pixi run -e py312-r pytest tests/test_errors.py::test_predict_fe_dtype_mismatch -q
```

### 2. `46b214dc` - Pin Formulaic To 1.2 Minor Line

Review:

- [`pyproject.toml`](pyproject.toml)

Questions:

- Is `formulaic>=1.2.0,<1.3` appropriate for code relying on 1.2 behavior?
- Is `pixi.lock` intentionally unchanged?

Suggested check:

```bash
pixi run -e py312-r python -c "import formulaic; print(formulaic.__version__)"
```

### 3. `c3861d15` - Add Formulaic Compatibility Smoke Tests

Review:

- [`tests/test_formulaic_compat.py`](tests/test_formulaic_compat.py)

Map tests to Formulaic concepts:

- MULTISTAGE `.deps` and `_hat`: parser links above.
- `encoder_state` shape: materializer write link above.
- `i()` state keys: pyfixest custom transform in
  [`factor_interaction.py`](pyfixest/estimation/formula/transforms/factor_interaction.py).
- FE state: pyfixest custom transform in
  [`fixed_effects_encoding.py`](pyfixest/estimation/formula/transforms/fixed_effects_encoding.py).
- Prediction round-trip: `ModelSpec.get_model_matrix`.

Questions:

- Do tests cover every undocumented/internal assumption?
- Would a Formulaic minor-version change fail loudly in CI?

Suggested check:

```bash
pixi run -e py312-r pytest tests/test_formulaic_compat.py -q
```

### 4. `4606275c` - Centralize Formulaic Internals

Review:

- [`formulaic_compat.py`](pyfixest/estimation/formula/formulaic_compat.py)
- Call sites:
  [`parse.py`](pyfixest/estimation/formula/parse.py),
  [`model_matrix.py`](pyfixest/estimation/formula/model_matrix.py),
  [`factor_interaction.py`](pyfixest/estimation/formula/transforms/factor_interaction.py),
  [`prediction.py`](pyfixest/estimation/post_estimation/prediction.py),
  [`feols_.py`](pyfixest/estimation/models/feols_.py)

Questions:

- Are direct Formulaic internals now behind named adapter helpers?
- Do helper names explain why each internal exists?
- Did behavior remain unchanged relative to the previous commit?

Formulaic cross-checks:

- `_flatten()` maps to `Structured._flatten`.
- `FactorValues.__wrapped__` maps to Formulaic's proxy wrapper.
- `_state` / `_spec` maps to stateful transform injection.
- `.deps` and `_hat` map to MULTISTAGE parser behavior.

Suggested check:

```bash
pixi run -e py312-r pytest tests/test_formula_parse.py tests/test_formulaic_compat.py -q
```

Note: `tests/test_predict_resid_fixef.py` may segfault during `rpy2` import in
this local environment; that failure happens before pyfixest code is collected.

### 5. `6d1d5e41` - Prefer Documented Categorical Metadata

Review:

- Documented categorical path in
  [`formulaic_compat.py`](pyfixest/estimation/formula/formulaic_compat.py)
- Remaining `i()` encoder-state path in the same file.

Formulaic cross-checks:

- `ModelSpec.factor_contrasts`
- `ModelSpec.factor_variables`

Questions:

- Does native `C()` unseen-category detection avoid raw `encoder_state`?
- Is raw `encoder_state` now limited to pyfixest's custom `i()` substate?
- Do binned `i()` terms still apply stored bin mappings before unseen checks?

Suggested check:

```bash
pixi run -e py312-r pytest \
  tests/test_others.py::test_predict_newdata_unseen_category \
  tests/test_others.py::test_predict_decoy_column_not_flagged_unseen \
  tests/test_others.py::test_predict_binned_i_not_flagged_unseen \
  tests/test_formulaic_compat.py -q
```

### 6. `e0dd2237` - Guard Formulaic Internal Assumptions

Review:

- `FormulaicCompatibilityError` in
  [`formulaic_compat.py`](pyfixest/estimation/formula/formulaic_compat.py)
- MULTISTAGE shape guard.
- `_hat` suffix guard.
- `encoder_state` shape guard.
- Guard tests in [`tests/test_formulaic_compat.py`](tests/test_formulaic_compat.py)

Questions:

- Would a changed `.deps` shape fail before IV parsing leaks wrong variables?
- Would a changed `_hat` convention fail before endogenous regressors leak into exogenous terms?
- Would a changed `encoder_state` shape fail before unseen categories are silently skipped?

Suggested check:

```bash
pixi run -e py312-r pytest tests/test_formulaic_compat.py -q
```

## Final Review Checklist

- `pixi.lock` is unchanged.
- `tests/test_formulaic_compat.py` passes.
- Targeted formula parse and prediction tests pass.
- Formulaic internals are greppable from one adapter file.
- Remaining known architectural follow-up: remove IV dependence on Formulaic
  MULTISTAGE entirely in a separate PR.
