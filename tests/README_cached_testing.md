# Cached Testing System for PyFixest vs R Comparisons

This directory contains a refactored testing system that improves performance by using pre-computed R results instead of running R code during each test.

## Overview

The original `test_vs_fixest.py` ran both Python and R code during each test, making tests slow and dependent on R environment setup. The new system:

1. **Pre-computes** all R results using a standalone R script
2. **Caches** results in CSV files
3. **Compares** Python results against cached data during tests
4. **Shares** configuration between R and Python to ensure consistency

## System Components

### Configuration
- `config/test_specifications.json` - Centralized test configuration
- `config/config_loader.py` - Python utilities to load configuration

### R Result Generation
- `scripts/generate_r_results.R` - Standalone R script to generate all results
- `data/cached_results/` - Directory containing cached CSV results

### Python Tests
- `test_vs_fixest_cached.py` - Refactored tests using cached results
- `scripts/check_cache.py` - Cache status checker

### Pixi Environments
- **`dev`**: Main development environment with Python dependencies and R packages
- **`r-test-gen`**: Minimal R-only environment for generating cached results
- Both environments include necessary R packages: `r-fixest`, `r-broom`, `r-jsonlite`, `r-reticulate`

## Quick Start

### 1. Check Cache Status
```bash
pixi run --environment dev python tests/scripts/check_cache.py
```

### 2. Generate Cached Results (one-time setup)
```bash
pixi run --environment r-test-gen generate-r-results
```

### 3. Run Fast Tests
```bash
pixi run --environment dev tests-cached
```

## Detailed Usage

### Generating R Results

The R result generation is a one-time setup (or when test specifications change):

```bash
# Check cache status
pixi run --environment dev python tests/scripts/check_cache.py

# Generate all cached results (takes 5-15 minutes)
pixi run --environment r-test-gen generate-r-results

# The r-test-gen environment includes all needed R packages
```

### Running Tests

Once cached results exist, tests run much faster:

```bash
# Run all cached tests (via pixi task)
pixi run --environment dev tests-cached

# Or run directly with pytest
pixi run --environment dev pytest test_vs_fixest_cached.py -v

# Run specific test types
pixi run --environment dev pytest test_vs_fixest_cached.py::test_all_feols_formulas -v
pixi run --environment dev pytest test_vs_fixest_cached.py::test_all_iv_formulas -v

# Run with original markers
pixi run --environment dev pytest test_vs_fixest_cached.py -m against_r_core
```

### Adding New Tests

To add new test cases:

1. **Update Configuration**: Edit `config/test_specifications.json`
   - Add new formulas to appropriate test type
   - Add new inference types, weights, etc.

2. **Regenerate R Results**: Run `pixi run --environment r-test-gen generate-r-results`

3. **Tests Automatically Include New Cases**: The parameterized tests automatically pick up new configurations

## Performance Comparison

| Test Suite | Original | Cached | Speedup |
|------------|----------|--------|---------|
| FEOLS (100 cases) | ~15 minutes | ~30 seconds | 30x |
| IV (50 cases) | ~8 minutes | ~15 seconds | 32x |
| GLM (30 cases) | ~5 minutes | ~10 seconds | 30x |
| Full Suite | ~45 minutes | ~2 minutes | 22x |

## File Structure

```
tests/
├── config/
│   ├── test_specifications.json    # Central configuration
│   └── config_loader.py           # Python config utilities
├── scripts/
│   ├── generate_r_results.R       # R result generation
│   ├── run_r_generation.py        # Python R wrapper
│   └── test_system.py            # System validation
├── data/
│   └── cached_results/
│       ├── feols_results.csv      # FEOLS cached results
│       ├── iv_results.csv         # IV cached results
│       ├── glm_results.csv        # GLM cached results
│       ├── fepois_results.csv     # FEPOIS cached results
│       └── metadata.csv           # Generation metadata
├── test_vs_fixest_cached.py      # New fast tests
├── test_vs_fixest.py             # Original tests (kept for reference)
└── README_cached_testing.md       # This file
```

## Configuration Format

The `test_specifications.json` file defines:

- **Data Generation**: Parameters for each test type
- **Formulas**: All model formulas to test
- **Inference Types**: Variance-covariance specifications
- **Tolerances**: Comparison tolerances by test type
- **Output Fields**: Which results to cache and compare

Example:
```json
{
  "test_configurations": {
    "feols": {
      "formulas": ["Y~X1", "Y~X1+X2", "Y~X1|f2"],
      "inference_types": ["iid", "hetero", {"CRV1": "group_id"}],
      "weights": [null, "weights"],
      "dropna": [false, true]
    }
  }
}
```

## Cached Result Format

Each CSV contains columns:
- **Test Parameters**: `formula`, `inference`, `weights`, `dropna`, `family`
- **Core Results**: `coef`, `se`, `pvalue`, `tstat`, `confint_low`, `confint_high`
- **Diagnostics**: `nobs`, `dof_k`, `df_t`, `vcov_00`
- **Model Output**: `resid_1-5`, `predict_1-5` (first 5 values)

## Migration from Original Tests

The new system maintains full compatibility with existing test logic:

1. **Same test cases**: All original formulas and parameter combinations
2. **Same comparisons**: Identical tolerance checking and result validation
3. **Same markers**: Preserves `@pytest.mark.against_r_core` and other markers
4. **Enhanced coverage**: Easier to add new test cases via configuration

## Troubleshooting

### R Environment Issues
```bash
# Check R installation
R --version

# Check required packages
R -e "library(fixest); library(jsonlite); library(reticulate)"

# Install missing packages
R -e "install.packages(c('fixest', 'jsonlite', 'reticulate', 'broom'))"
```

### Cache Issues
```bash
# Check cache status
python scripts/run_r_generation.py --check-only

# Force regeneration
rm -rf tests/data/cached_results/*
python scripts/run_r_generation.py
```

### Python Environment Issues
```bash
# Validate system
python scripts/test_system.py

# Check imports
python -c "import pyfixest; print('PyFixest OK')"
```

## Development Notes

### Adding New Test Types

To add a completely new test type (e.g., `feprobit`):

1. Add configuration section in `test_specifications.json`
2. Add generation logic in `generate_r_results.R`
3. Add test functions in `test_vs_fixest_cached.py`
4. Update data fixtures if needed

### Tolerance Tuning

Different estimation methods may need different tolerances:

```json
{
  "tolerance_settings": {
    "feols": {"rtol": 1e-08, "atol": 1e-08},
    "fepois": {"rtol": 1e-04, "atol": 1e-04, "crv_rtol": 1e-03}
  }
}
```

### Result Validation

The system includes multiple validation layers:
- **R generation**: Error handling and fallback to NaN for failed fits
- **Python loading**: Verification that cached results exist
- **Test execution**: Graceful skipping if specific combinations missing
