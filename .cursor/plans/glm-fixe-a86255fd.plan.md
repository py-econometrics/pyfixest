<!-- a86255fd-e559-425e-82e2-e2ad35c50047 460c8778-a2e6-429c-924f-1f88709080a0 -->
# Extend Fixed Effects Support to feglm (Logit/Binomial Family)

## Current State - Gaussian Implementation Complete

Fixed effects support for Gaussian GLMs is **fully working**:

- ✅ Backend infrastructure in place (`demeaner_backend` parameter)
- ✅ Proper variance calculation using `self._df_t`
- ✅ All tests passing (30/30): internal, R comparison, and error handling
- ✅ Results match R's `fixest::feglm(family=gaussian())` exactly

**Restriction in place** (line 102-103 in `feglm_.py`):

```python
if self._fe is not None and self._method != "feglm-gaussian":
    raise NotImplementedError("Fixed effects are not yet supported for GLMs.")
```

## Logit Family Structure

The `Felogit` class exists in `felogit_.py` and:

- Inherits from `Feglm` (like `Fegaussian`)
- Implements binomial family with logit link
- Has `_method = "feglm-logit"`
- Uses `_vcov_iid()` from `Feglm` (returns `self._bread`, no dispersion multiplier needed)
- **Missing**: `demeaner_backend` parameter (not passed to parent)

**Key difference from Gaussian**: Logit uses dispersion φ = 1.0, so no sigma² multiplier in variance calculation (already correct in base `Feglm._vcov_iid()`).

## Implementation Steps

### 1. Update Felogit to Accept demeaner_backend Parameter

Modify `pyfixest/estimation/felogit_.py`:

**a) Add import** (after line 8):

```python
from pyfixest.estimation.literals import DemeanerBackendOptions
```

**b) Add parameter to `__init__`** (after line 35):

```python
demeaner_backend: DemeanerBackendOptions = "numba",
```

**c) Pass to parent `super().__init__`** (after line 57):

```python
demeaner_backend=demeaner_backend,
```

### 2. Update Fixed Effects Restriction in feglm_.py

Modify line 102 in `pyfixest/estimation/feglm_.py`:

**Change from:**

```python
if self._fe is not None and self._method != "feglm-gaussian":
```

**To:**

```python
if self._fe is not None and self._method not in ["feglm-gaussian", "feglm-logit"]:
```

This allows both Gaussian and Logit, while still blocking Probit.

### 3. Add Tests for Logit with Fixed Effects

**a) Update test formulas in `test_vs_fixest.py`:**

Currently `glm_fmls_with_fe` is only used for Gaussian. Update the test to include logit:

**Modify `test_glm_with_fe_vs_fixest`** (around line 901):

- Add `@pytest.mark.parametrize("family", ["gaussian", "logit"])`
- Update R model creation to handle both families:
  ```python
  if family == "gaussian":
      fit_r = fixest.feglm(ro.Formula(r_fml), data=data_r, family=stats.gaussian(), vcov=r_inference)
  elif family == "logit":
      fit_r = fixest.feglm(ro.Formula(r_fml), data=data_r, family=stats.binomial(link="logit"), vcov=r_inference)
  ```


**b) Update `test_errors.py`:**

Modify `test_glm_errors` (around line 820):

- Update to test that logit now allows fixed effects
- Keep probit blocked

### 4. Validation

Run tests to ensure:

- Logit with single FE: `Y ~ X1 | f1` ✓
- Logit with multiple FE: `Y ~ X1 | f1 + f2` ✓
- Logit with multiple covariates: `Y ~ X1 + X2 | f2` ✓
- All inference types: `iid`, `hetero`, `CRV1` ✓
- Results match R's `fixest::feglm(family=binomial(link="logit"))` ✓
- IRLS converges properly with fixed effects

## Key Files to Modify

- `pyfixest/estimation/felogit_.py` (add demeaner_backend parameter)
- `pyfixest/estimation/feglm_.py` (update restriction to include logit)
- `tests/test_vs_fixest.py` (extend tests to logit family)
- `tests/test_errors.py` (update error test for logit)

## Expected Behavior

After implementation:

1. `pf.feglm(fml="Y ~ X1 | f1", family="logit", demeaner_backend="rust")` works
2. Results match R's `fixest::feglm(family=binomial(link="logit"))`
3. IRLS with fixed effects converges properly
4. Variance calculation uses `self._bread` (no dispersion multiplier, φ=1)
5. Probit remains blocked (for now)

## Notes

- Logit doesn't need a custom `_vcov_iid()` method (unlike Gaussian) because dispersion φ = 1.0
- The IRLS algorithm in `Feglm.get_fit()` already handles demeaning at each iteration via `residualize()`
- Main differences from Gaussian: link function, variance function, and convergence behavior
- Separation checks are already in place and conditional on fixed effects

### To-dos

- [ ] Add DemeanerBackendOptions import to felogit_.py
- [ ] Add demeaner_backend parameter to Felogit.__init__ and pass to parent
- [ ] Update feglm_.py restriction to allow both gaussian and logit families
- [ ] Extend test_glm_with_fe_vs_fixest to test both gaussian and logit families
- [ ] Update test_glm_errors to allow logit with fixed effects
- [ ] Run tests and validate logit with FE matches R fixest results
