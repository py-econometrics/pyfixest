# Pyfixest Test Refactoring Framework

This directory contains the refactored test framework for comparing pyfixest against R's fixest package.

## Structure

- `config/` - Test parameter definitions and generators
  - `base/` - Shared base classes and utilities
  - `feols/` - FEOLS-specific test configurations
- `r_cache/` - R test execution and caching
  - `base/` - Shared R execution utilities
  - `feols/` - FEOLS-specific R scripts and runners
- `tests/` - Python test files that compare against cached R results
- `data/cached_results/` - Cached R test results organized by method
- `manage_cache.py` - Unified cache management CLI

## Migration Status

- ✅ `test_single_fit_feols` - Completed
- ⏳ `test_single_fit_fepois` - Planned
- ⏳ `test_single_fit_iv` - Planned

## Usage

### Unified Commands (Recommended)

```bash
# List all available test methods
pixi run test-cache-all-methods

# Generate cached R results for all implemented methods
pixi run test-cache-all-results

# Run all cached tests
pixi run test-run-all-cached

# Show cache summary for all methods
pixi run test-cache-all-summary

# Complete workflow (generate + test)
pixi run test-cached-workflow-all
```

### Method-Specific Commands

```bash
# Generate cache for specific method only
cd tests && python refactor/cache_manager.py generate feols

# Show summary for specific method
cd tests && python refactor/cache_manager.py summary feols

# Clear cache for specific method
cd tests && python refactor/cache_manager.py clear feols
```

## Development

```bash
# Work on specific method
pixi run test-cache-feols-results
pixi run test-run-feols-cached

# Cache management
pixi run test-cache-all-summary
pixi run test-cache-all-clear
```
