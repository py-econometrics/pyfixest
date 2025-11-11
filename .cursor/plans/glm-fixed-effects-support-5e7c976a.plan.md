<!-- 5e7c976a-5d57-4559-9eec-e537dca963a1 9818a1cd-a593-4f12-a9f7-6caba18c5fcb -->
# Add Fixed Effects Support to feglm (Gaussian Family)

## Current State

The `Feglm` class (in `feglm_.py`) currently raises `NotImplementedError` at line 99-100 when fixed effects are present. However, the infrastructure is largely in place:

- The `residualize()` method exists (lines 304-323) and correctly calls `demean()`
- Fixed effects are already handled in `prepare_model_matrix()` and `to_array()` methods
- Separation checks (lines 102-126) are already conditional on `self._fe is not None`

The `Fepois` class successfully implements fixed effects in its IRLS algorithm (lines 282-294 in `fepois_.py`) by calling `demean()` with fixed effects and weights at each iteration.

## Implementation Steps

### 1. Remove Fixed Effects Restriction in feglm_.py

Remove the restriction in `prepare_model_matrix()` method:

- Delete lines 99-100 that raise `NotImplementedError` for fixed effects
- This will allow the rest of the code to execute when fixed effects are present

### 2. Verify residualize() Method Integration

The `residualize()` method (lines 304-323) already properly handles fixed effects:

- Returns unchanged data when `flist is None` (no fixed effects)
- Calls `demean()` with weights when fixed effects exist
- This method is already called in `get_fit()` at line 182

The implementation should work as-is since:

- `Fegaussian` inherits from `Feglm`
- The IRLS algorithm in `get_fit()` calls `residualize()` at each iteration (line 182)
- Weights are properly passed as `W_tilde.flatten()` (line 186)

### 3. Test the Implementation

Enable and extend existing tests:

**a) Enable skipped test in `test_feols_feglm_internally.py`:**

- Remove the `@pytest.mark.skip` decorator on line 60
- Run the test to verify it passes with the implementation

**b) Add basic R comparison tests:**

- Add tests to `test_vs_fixest.py` comparing `pf.feglm(family="gaussian")` against R's `fixest::feglm(family="gaussian")`
- Test formulas: `"Y ~ X1 | f1"`, `"Y ~ X1 | f1 + f2"`, `"Y ~ X1 + X2 | f2"`
- Compare: coefficients, standard errors, residuals
- Use existing test patterns from the `glm_fmls` tests (around line 119)

### 4. Validation Checklist

Ensure the implementation correctly handles:

- Single fixed effect: `Y ~ X1 | f1`
- Multiple fixed effects: `Y ~ X1 | f1 + f2`
- Fixed effects with multiple covariates: `Y ~ X1 + X2 | f1`
- Proper weight handling in demeaning at each IRLS iteration
- Convergence behavior (should still converge in 1 iteration for Gaussian)

## Key Files to Modify

- `pyfixest/estimation/feglm_.py` (remove restriction)
- `tests/test_feols_feglm_internally.py` (enable skipped test)
- `tests/test_vs_fixest.py` (add R comparison tests)

## Expected Behavior

After implementation, `pf.feglm(fml="Y ~ X1 | f1", family="gaussian")` should:

1. Accept the formula without raising `NotImplementedError`
2. Demean variables by fixed effects at each IRLS iteration
3. Produce identical results to R's `fixest::feglm()` with Gaussian family
4. For Gaussian family, should effectively match `feols()` results (with appropriate inference adjustments)

### To-dos

- [ ] Remove NotImplementedError for fixed effects in feglm_.py (lines 99-100)
- [ ] Remove @pytest.mark.skip decorator from test_feols_feglm_internally
- [ ] Add R comparison tests for Gaussian GLM with fixed effects to test_vs_fixest.py
- [ ] Run tests and validate against R fixest results
