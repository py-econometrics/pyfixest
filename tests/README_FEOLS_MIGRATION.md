# FEOLS Test Migration Framework

This document explains the new cached testing framework for migrating `test_single_fit_feols` to use cached R results.

## Overview

The framework consists of:

1. **Abstract Base Classes** (`config/test_cases.py`) - Define test case structure
2. **FEOLS Test Cases** (`config/feols_tests.py`) - `TestSingleFitFeols` implementation
3. **Test Generator** (`config/feols_test_generator.py`) - Generates all parameter combinations
4. **R Cache Runner** (`r_cache/r_test_runner.py`) - Runs R tests and caches results
5. **Cached Tests** (`test_feols_cached.py`) - New parametrized tests using cached R results

## Key Benefits

- **Speed**: R tests run once, Python tests compare against cached results
- **Reproducibility**: Hash-based caching ensures consistency
- **Maintainability**: Clear separation between test configuration and execution
- **Type Safety**: Abstract base classes with validation
- **Human Readable**: JSON cache files can be inspected and debugged easily
- **Language Agnostic**: R writes JSON directly, no Python dependency

## Usage

### Using Pixi Tasks (Recommended)

```bash
# 1. Generate test hyperparameters (shows how many test cases will be created)
pixi run test-generate-params

# 2. Run R scripts and cache results (time-intensive, run once)
pixi run test-cache-r-results

# 3. Run fast Python tests against cached results
pixi run test-run-cached

# 4. Complete workflow (all three steps above)
pixi run test-cached-workflow

# 5. Management tasks
pixi run test-cache-summary    # Show cache status
pixi run test-cache-clear      # Clear all cached results
```

### Manual Usage (Alternative)

```bash
# Generate all R results and cache them
cd tests
python manage_feols_cache.py generate

# Check cache status
python manage_feols_cache.py summary

# Run all cached FEOLS tests
pytest test_feols_cached.py -v

# Run specific test cases
pytest test_feols_cached.py::test_feols_vs_cached_r -k "feols_00001"

# Clear cache
python manage_feols_cache.py clear

# Force refresh cache
python manage_feols_cache.py generate --force
```

## Framework Structure

### Test Case Definition

```python
# Example test case for test_single_fit_feols migration
test_case = TestSingleFitFeols(
    test_id="feols_001",
    formula="Y~X1+X2",
    inference="hetero",
    weights="weights",
    dropna=True,
    f3_type="categorical",
    demeaner_backend="numba"
)
```

### Naming Convention

Test case classes follow the pattern `Test{OriginalFunctionName}`:
- `TestSingleFitFeols` → migrates `test_single_fit_feols`
- `TestSingleFitFepois` → will migrate `test_single_fit_fepois`
- `TestSingleFitIv` → will migrate `test_single_fit_iv`
- etc.

This naming helps track which original test function each test case class replaces.

### Parameter Combinations

The framework generates all parameter combinations from the original `test_single_fit_feols`:

- **Formulas**: All OLS formulas + OLS-but-not-Poisson formulas
- **Inference**: "iid", "hetero", {"CRV1": "group_id"}
- **Weights**: None, "weights"
- **Dropna**: False, True
- **F3 Types**: "str", "object", "int", "categorical", "float"
- **Backends**: "numba", "jax", "rust"

### Validation

Each test case validates its parameters:
- JAX/Rust backends only support string f3_type
- Cluster variables must exist in data
- All numeric parameters must be positive

## Migration Path

1. **Phase 1** (Current): Framework setup with FEOLS tests
2. **Phase 2**: Extend to FEPOIS, FEIV, DID, etc.
3. **Phase 3**: Gradually replace original tests
4. **Phase 4**: Remove old test infrastructure

## Files Created

```
tests/
├── config/
│   ├── __init__.py
│   ├── test_cases.py           # Abstract base classes
│   ├── test_registry.py        # Test registry
│   ├── feols_tests.py          # TestSingleFitFeols class
│   └── feols_test_generator.py # Parameter combination generator
├── r_cache/
│   ├── __init__.py
│   └── r_test_runner.py        # R test execution and caching
├── test_feols_cached.py        # New cached tests
├── manage_feols_cache.py       # Cache management script
└── README_FEOLS_MIGRATION.md   # This file
```

## Example Workflow

```python
# 1. Generate test cases for test_single_fit_feols migration
from config.feols_test_generator import generate_feols_test_cases
test_cases = generate_feols_test_cases()
print(f"Generated {len(test_cases)} TestSingleFitFeols test cases")

# 2. Run R tests and cache (one-time)
from r_cache.r_test_runner import FeolsRTestRunner
runner = FeolsRTestRunner()
results = runner.run_all_tests(test_cases)

# 3. Run Python tests against cache (fast)
pytest test_feols_cached.py
```

## Pixi Tasks Reference

| Task | Description | Usage |
|------|-------------|-------|
| `test-generate-params` | Generate and count test hyperparameters | Shows how many test cases will be created |
| `test-cache-r-results` | Run R scripts and cache all results | Time-intensive, run once or when parameters change |
| `test-run-cached` | Run Python tests against cached R results | Fast execution, main testing command |
| `test-cached-workflow` | Complete workflow (all three steps) | One-command setup for new environments |
| `test-cache-summary` | Show cache status and statistics | Check what's cached and when |
| `test-cache-clear` | Clear all cached R results | Clean slate for regeneration |

## Next Steps

1. Test the framework with a subset of cases: `pixi run test-generate-params`
2. Generate initial cache: `pixi run test-cache-r-results`
3. Run fast tests: `pixi run test-run-cached`
4. Extend to other test types (FEPOIS, FEIV, etc.)
5. Add more sophisticated caching strategies
6. Integrate with CI/CD pipeline
