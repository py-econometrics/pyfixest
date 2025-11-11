<!-- a86255fd-e559-425e-82e2-e2ad35c50047 f5adc961-97fa-4a2f-b1f7-c67734af5cd0 -->
# Add Fixed Effects Support to feglm (Gaussian Family)

## Current State

The `Feglm` class currently raises `NotImplementedError` at line 99-100 when fixed effects are present. Additionally, `Feglm` hardcodes the numba demeaning backend instead of using the configurable backend system that `feols` uses.

**Key differences from feols:**

- `Feglm` imports `demean` directly from `demean_.py` (line 11) - this is the numba backend
- `Feglm` calls this hardcoded `demean` function in `residualize()` (line 317)
- `Feols` uses a `demeaner_backend` parameter and selects the demean function from the `BACKENDS` dictionary
- `Fepois` tries to pass `demeaner_backend` to `Feglm.__init__`, but `Feglm` doesn't accept it

## Implementation Steps

### 1. Update Feglm to Use Backend Infrastructure

Modify `pyfixest/estimation/feglm_.py`:

**a) Update imports** (line 11):

- Remove: `from pyfixest.estimation.demean_ import demean`
- Add: `from pyfixest.estimation.backends import BACKENDS`
- Add: `from pyfixest.estimation.literals import DemeanerBackendOptions`

**b) Add `demeaner_backend` parameter to `__init__`** (line 21):

- Add `demeaner_backend: DemeanerBackendOptions = "numba"` parameter
- Store as `self._demeaner_backend = demeaner_backend`
- Set `self._demean_func = BACKENDS[demeaner_backend]["demean"]` (similar to how `Feols` does it at line 323-327 of `feols_.py`)

**c) Update `residualize()` method** (line 317):

- Replace `demean(...)` with `self._demean_func(...)`

### 2. Remove Fixed Effects Restriction

In `pyfixest/estimation/feglm_.py`:

- Delete lines 99-100 that raise `NotImplementedError` for fixed effects

### 3. Update Fegaussian Constructor

Modify `pyfixest/estimation/fegaussian_.py`:

- Add `demeaner_backend: DemeanerBackendOptions = "numba"` parameter to `__init__` (around line 29)
- Pass it to the parent `Feglm.__init__` call

### 4. Test the Implementation

**a) Enable skipped test in `test_feols_feglm_internally.py`:**

- Remove the `@pytest.mark.skip` decorator on line 60

**b) Add R comparison tests in `test_vs_fixest.py`:**

- Test formulas: `"Y ~ X1 | f1"`, `"Y ~ X1 | f1 + f2"`, `"Y ~ X1 + X2 | f2"`
- Compare: coefficients, standard errors, residuals against R's `fixest::feglm(family="gaussian")`

### 5. Validation

Ensure the implementation correctly handles:

- Backend selection (numba, rust, jax) works correctly
- Single and multiple fixed effects
- Proper weight handling in demeaning at each IRLS iteration
- Convergence (should still converge in 1 iteration for Gaussian)

## Key Files to Modify

- `pyfixest/estimation/feglm_.py` (add backend infrastructure, remove restriction)
- `pyfixest/estimation/fegaussian_.py` (pass demeaner_backend parameter)
- `tests/test_feols_feglm_internally.py` (enable skipped test)
- `tests/test_vs_fixest.py` (add R comparison tests)

## Expected Behavior

After implementation:

1. Users can specify `demeaner_backend` for `feglm`, just like with `feols`
2. `pf.feglm(fml="Y ~ X1 | f1", family="gaussian", demeaner_backend="rust")` works with any backend
3. Results match R's `fixest::feglm()` with Gaussian family
4. For Gaussian family, effectively matches `feols()` results (with appropriate inference adjustments)

### To-dos

- [ ] Update imports in feglm_.py to use BACKENDS and DemeanerBackendOptions
- [ ] Add demeaner_backend parameter to Feglm.__init__ and set self._demean_func
- [ ] Update residualize() method to use self._demean_func instead of hardcoded demean
- [ ] Remove NotImplementedError for fixed effects in feglm_.py
- [ ] Add demeaner_backend parameter to Fegaussian.__init__ and pass to parent
- [ ] Remove @pytest.mark.skip decorator from test in test_feols_feglm_internally.py
- [ ] Add R comparison tests to test_vs_fixest.py
- [ ] Run tests and validate against R fixest results
