# CuPy-based FWL Demeaner Implementation

## Summary

Successfully implemented a GPU-accelerated demeaning backend for pyfixest using CuPy and the Frisch-Waugh-Lovell (FWL) theorem.

## What Was Built

### 1. Core Implementation (`pyfixest/estimation/cupy/demean_cupy_.py`)

**CupyFWLDemeaner Class**:
- Implements FWL theorem: demean by solving `D @ theta = x` where D is the sparse fixed effects dummy matrix
- **Hybrid solving strategy**:
  - **Normal equations**: `(D'D) @ theta = D'x` with `spsolve` (vectorized, fast for small problems)
  - **LSMR fallback**: Solves `min ||Dx - b||²` directly (more stable for large/ill-conditioned problems)
- **Auto-detection**: Automatically switches between GPU (CuPy) and CPU (scipy) based on availability
- **Sparse matrix construction**: Efficient COO→CSR conversion from integer-coded fixed effects
- **Full dummy matrix**: Creates ALL FE levels (no level dropping) to match alternating projections behavior
- **Weighted demeaning**: Supports observation weights via WLS transform
- **Operator caching**: Caches sparse FE matrices to amortize construction cost

**Key Features**:
- GPU acceleration via CuPy (NVIDIA CUDA)
- Graceful CPU fallback when GPU unavailable
- Configurable thresholds for normal equations vs. LSMR
- Compatible with existing pyfixest demean signature

### 2. Backend Integration

**Files Modified**:
- `pyfixest/estimation/backends.py`: Added `"cupy"` backend with conditional imports
- `pyfixest/estimation/literals.py`: Added `"cupy"` to `DemeanerBackendOptions`
- `pyfixest/estimation/demean_.py`: Updated `_set_demeaner_backend()` to handle cupy

**Usage**:
```python
import pyfixest as pf

# Use CuPy backend (auto-detects GPU)
model = pf.feols("y ~ x1 + x2 | firm + year", data=data, demeaner_backend="cupy")

# Or use directly
from pyfixest.estimation.cupy.demean_cupy_ import CupyFWLDemeaner

demeaner = CupyFWLDemeaner(use_gpu=True)
x_demeaned, success = demeaner.demean(x, flist, weights)
```

### 3. Testing Infrastructure

**Test File**: `tests/test_demean.py`

**Added Tests**:
1. `test_cupy_vs_rust_consistency()`: Verify CuPy produces same results as Rust backend
2. `test_cupy_cpu_fallback()`: Test graceful CPU fallback
3. `test_cupy_weighted_demeaning()`: Test weighted case
4. `test_cupy_sparse_matrix_caching()`: Test operator caching
5. `test_cupy_full_dummy_matrix()`: Verify no level dropping
6. `test_cupy_normal_equations_vs_lsmr()`: Test both solving strategies

**Updated Tests**:
- Added `demean_cupy` to parametrized tests (`test_demean`, `test_demean_complex_fixed_effects`)
- Updated `test_set_demeaner_backend()` to include cupy case
- All CuPy tests conditionally skipped if CuPy not installed

### 4. Environment Configuration

**File**: `pixi.toml`

**Changes**:
1. Added `cupy-cuda12x >= 13.0.0` to `[feature.dev.pypi-dependencies]`
2. Created new `[feature.cupy]` environment (similar to jax)
3. Added `cupy` to `[environments]` list

**Installation**:
```bash
# Install with CuPy support (pip)
pip install pyfixest[gpu]

# Or use pixi environment
pixi shell -e cupy
```

### 5. Dependencies

**File**: `pyproject.toml`

**Changes**:
1. Added `[project.optional-dependencies]` section with GPU extras:
   ```toml
   gpu = ["cupy-cuda12x>=13.0.0"]
   ```
2. Updated mypy overrides to ignore cupy/cupyx imports

## Technical Details

### Solving Strategy Decision Tree

```
Problem Size: n_fe_params
        |
        v
n_fe_params < 10,000?
        |
    Yes |   No
        |    └──> Use LSMR (stable)
        v
Compute D'D density
        |
        v
density < 0.5?
        |
    Yes |   No
        |    └──> Use LSMR (D'D too dense)
        v
Try Normal Equations
        |
Success? Yes ──> Return result (FAST!)
        |
        No
        |
        v
Fall back to LSMR
```

### Why Normal Equations Are Stable for Dummy Matrices

**Your excellent point**: D contains only 0s and 1s, so computing D'D is numerically stable:
- No rounding errors in matrix multiplication
- D'D entries are non-negative integers (counts/co-occurrences)
- For typical FE structures (not nested), D'D is well-conditioned

**When Normal Equations Fail**:
1. **Nested FEs**: D'D is singular (not full rank)
2. **Very dense D'D**: Memory explosion (GB+ on GPU)
3. **Highly imbalanced groups**: Large condition number

**Solution**: Hybrid approach tries normal equations first, falls back to LSMR if needed.

### Sparse Matrix Construction

Creates **full dummy matrix** (no level dropping):
```python
for each FE dimension j:
    n_groups = max(flist[:, j]) + 1  # ALL groups
    Create COO arrays: (row_indices, col_indices, data=1.0)

Concatenate all FE dimensions
Convert to CSR format (efficient for matrix-vector products)
```

This matches the behavior of:
- Alternating projections (existing backends)
- `fixef()` method with `"-1+"` formula
- pyhdfe library

## Performance Characteristics

| Problem Size | Strategy | Speed | Memory |
|-------------|----------|-------|--------|
| **Small** (n_fe < 5k) | Normal eq (vectorized) | ⚡⚡⚡ | Low |
| **Medium** (5k < n_fe < 10k) | Normal eq or LSMR | ⚡⚡ | Medium |
| **Large** (n_fe > 10k) | LSMR (sequential) | ⚡ | Low |
| **Very Large** (GPU, n_fe > 100k) | GPU LSMR | ⚡⚡ | High |

**Bottleneck**: Loop over columns for LSMR (scipy.sparse.linalg.lsmr doesn't support 2D RHS)

**Future Optimization**: If sparse QR becomes available, could vectorize across columns.

## Files Created

```
pyfixest/estimation/cupy/
├── __init__.py
└── demean_cupy_.py

tests/
└── test_demean.py (modified)

CUPY_IMPLEMENTATION.md (this file)
test_cupy_basic.py (basic test script)
```

## Files Modified

```
pyfixest/estimation/
├── backends.py
├── literals.py
└── demean_.py

pyproject.toml
pixi.toml
```

## Testing Instructions

### Basic Test (CPU only)
```bash
python3 test_cupy_basic.py
```

### Run Unit Tests
```bash
# With CuPy installed
pytest tests/test_demean.py -v -k cupy

# Without CuPy (tests should be skipped)
pytest tests/test_demean.py -v
```

### Full Test Suite
```bash
pixi run -e dev tests
```

## Next Steps

1. **Install CuPy in dev environment**:
   ```bash
   pixi install -e dev
   # or
   pixi shell -e dev
   pip install cupy-cuda12x  # Already in pixi.toml
   ```

2. **Run tests**:
   ```bash
   python3 test_cupy_basic.py
   pytest tests/test_demean.py::test_cupy_vs_rust_consistency -v
   ```

3. **Benchmark**:
   ```bash
   pytest tests/test_demean.py::test_demean_complex_fixed_effects --benchmark-only
   ```

4. **Documentation** (optional):
   - Add CuPy backend to user documentation
   - Add GPU acceleration guide
   - Performance comparison table

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **GPU Library** | CuPy | Native CUDA, scipy-compatible API |
| **Solver** | LSMR + Normal Equations | Hybrid: fast for small, stable for large |
| **FE Encoding** | Keep integer codes | Maintains API compatibility |
| **Sparse Format** | CSR | Efficient for matrix-vector products |
| **Level Dropping** | None (full dummy matrix) | Matches existing backends |
| **Parallelization** | Sequential loop | LSMR only supports 1D RHS |
| **Caching** | Hash-based operator cache | Amortize sparse matrix construction |
| **Fallback** | Automatic CPU fallback | Graceful degradation |

## Known Limitations

1. **No column vectorization**: LSMR only accepts 1D right-hand side, so we loop over columns
2. **CuPy lsqr incompatibility**: CuPy's `lsqr` requires square matrices (normal equations), different from scipy
3. **CUDA version dependency**: Requires matching CUDA toolkit version (12.x or 11.x)
4. **No macOS GPU support**: CUDA is NVIDIA-only (macOS uses Metal/OpenCL)

## References

- **FWL Theorem**: https://en.wikipedia.org/wiki/Frisch–Waugh–Lovell_theorem
- **CuPy Documentation**: https://docs.cupy.dev/
- **scipy.sparse.linalg.lsmr**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
- **pyfixest fixest**: https://github.com/lrberge/fixest (R implementation)
