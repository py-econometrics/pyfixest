# Optimization Specification: demean_accelerated.rs

## 1. Current Implementation Analysis

### 1.1 Overview of demean_accelerated.rs

The current implementation in `src/demean_accelerated.rs` (336 lines) provides:

- **Irons-Tuck acceleration**: Applied every 3rd iteration
- **Struct abstractions**: `FactorDemeaner`, `MultiFactorDemeaner`, `AccelerationBuffers`, `IronTucksAcceleration`
- **Parallelization**: rayon for column-level parallelism
- **Memory**: Heap-allocated `Vec<f64>` buffers

### 1.2 Comparison: demean.rs vs demean_accelerated.rs

| Aspect | demean.rs | demean_accelerated.rs |
|--------|-----------|----------------------|
| Algorithm | Simple alternating projection | Irons-Tuck acceleration |
| Iteration | One projection per iter | 2 projections + acceleration step |
| Memory | Minimal buffers | 6 buffers × n_samples |
| Convergence | Element-wise SAD | Element-wise SAD |

### 1.3 Reference: fixest C++ (demeaning.cpp)

Key features in fixest not present in current Rust implementation:

| Feature | fixest | demean_accelerated.rs |
|---------|--------|----------------------|
| Grand acceleration | ✓ (3-point history) | ✗ |
| 2-FE optimization | ✓ (no N-length temps) | ✗ |
| SSR convergence | ✓ (every 40 iters) | ✗ |
| Coefficient-based | ✓ (iterates on FE coeffs) | ✗ (observation-based) |

---

## 2. Missing Parts (vs fixest)

### 2.1 Grand Acceleration (Priority: HIGH)

fixest implements a **two-tier acceleration scheme**:

```
Standard iterations: Apply Irons-Tuck every 3 iterations
Grand acceleration: Every `iter_grandAcc` iterations, apply Irons-Tuck
                    on a 3-point history (Y, GY, GGY) of coefficient vectors
```

The grand acceleration operates on a coarser timescale, accelerating convergence on slow-moving modes. This can significantly reduce iteration count for hard-to-converge problems.

**Implementation sketch:**
```rust
struct GrandAccelerationState {
    y: Vec<f64>,      // First history point
    gy: Vec<f64>,     // Second history point
    ggy: Vec<f64>,    // Third history point
    counter: usize,   // Cycles 0-2
    interval: usize,  // Apply every N iterations (default ~15)
}
```

### 2.2 Specialized 2-FE Path (Priority: MEDIUM)

When `n_factors == 2`, fixest uses a specialized routine that:
- Stores second FE coefficients in a `nb_coef_Q[1]`-length buffer instead of `n_obs`
- Avoids materializing full N-length residual vectors
- Alternates between updating both effects without intermediate storage

Current implementation always allocates `n_samples`-length buffers regardless of factor count.

### 2.3 SSR-Based Convergence (Priority: MEDIUM)

fixest checks residual sum-of-squares every 40 iterations:

```cpp
ssr = Σ(input[i] - mu_current[i])²
if (stopping_crit(ssr_old, ssr, diffMax)) break;
```

This complements the element-wise convergence check and can detect convergence earlier in some cases.

### 2.4 Coefficient-Based Iteration (Priority: LOW)

fixest iterates on FE **coefficients** rather than demeaned **observations**:
- Coefficient vector length: `Σ n_groups[j]` (often << n_samples)
- More cache-friendly for problems with many observations but few groups
- Requires restructuring the core algorithm

---

## 3. Potential Speedup Opportunities

### 3.1 SIMD Vectorization (Priority: HIGH)

Current inner loops rely on compiler autovectorization:

```rust
// Current: relies on autovectorization
for i in 0..n {
    self.buffers.delta_gx[i] = self.buffers.ggx_curr[i] - gx_tmp;
    // ...
}
```

**Opportunity**: Use explicit SIMD via `std::simd` (nightly) or `wide` crate:

```rust
use wide::f64x4;

// Process 4 elements at a time
for chunk in buffers.chunks_exact_mut(4) {
    let a = f64x4::from_slice(a_slice);
    let b = f64x4::from_slice(b_slice);
    (a - b).store(chunk);
}
```

Potential gains:
- **2-4x** for memory-bound operations (likely scenario)
- Requires careful handling of non-aligned tails

### 3.2 Memory Layout Optimization (Priority: HIGH)

Current: Separate `Vec<f64>` for each buffer (AoS pattern)

```rust
struct AccelerationBuffers {
    x_curr: Vec<f64>,
    gx_curr: Vec<f64>,
    ggx_curr: Vec<f64>,
    // ... 6 separate allocations
}
```

**Opportunity**: Interleaved SoA layout for better cache locality:

```rust
struct InterleavedBuffers {
    // All data in single allocation, interleaved for spatial locality
    data: Vec<f64>,  // [x0, gx0, ggx0, x1, gx1, ggx1, ...]
}
```

Or single contiguous allocation with computed offsets:

```rust
struct AccelerationBuffers {
    data: Vec<f64>,  // Single allocation: 6 * n_samples
    n_samples: usize,
}
impl AccelerationBuffers {
    fn x_curr(&mut self) -> &mut [f64] { &mut self.data[0..self.n_samples] }
    // ...
}
```

### 3.3 Reduce Per-Column Allocations (Priority: HIGH)

Current implementation allocates `MultiFactorDemeaner` per column:

```rust
// src/demean_accelerated.rs:274
let process_column = |(k, mut col): (...)| {
    let demeaner = MultiFactorDemeaner::new(...);  // Allocation per column!
    let mut acceleration = IronTucksAcceleration::new(...);
    // ...
};
```

**Opportunity**: Pre-allocate demeaners and reuse via thread-local storage:

```rust
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    static DEMEANER: RefCell<Option<MultiFactorDemeaner>> = RefCell::new(None);
}

// Or use rayon's broadcast for pre-allocation
```

### 3.4 Convergence Check Optimization (Priority: MEDIUM)

Current: Full pass over all elements every iteration:

```rust
fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
}
```

**Opportunity**: Early exit with SIMD max-reduction:

```rust
fn sad_converged_simd(a: &[f64], b: &[f64], tol: f64) -> bool {
    // SIMD: compute max |a-b| in chunks, early exit if any chunk exceeds tol
    let tol_vec = f64x4::splat(tol);
    for (a_chunk, b_chunk) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let diff = (f64x4::from_slice(a_chunk) - f64x4::from_slice(b_chunk)).abs();
        if diff.reduce_max() >= tol {
            return false;
        }
    }
    // Handle remainder...
    true
}
```

### 3.5 Group Mean Computation (Priority: MEDIUM)

Current scatter-gather pattern:

```rust
// Scatter: accumulate weighted sums
input.iter().zip(&self.sample_weights).zip(&self.group_ids)
    .for_each(|((&xi, &wi), &gid)| {
        self.group_weighted_sums[gid] += wi * xi;  // Random access
    });
```

**Opportunity**:
- Sort observations by group ID for sequential access (one-time cost)
- Use sparse matrix representation for very large groups
- Consider prefix sums for sorted data

### 3.6 Use ndarray-linalg for BLAS (Priority: LOW)

Add `ndarray-linalg` for optimized linear algebra:

```toml
[dependencies]
ndarray-linalg = { version = "0.16", features = ["openblas-system"] }
```

Could accelerate matrix operations if algorithm is restructured.

---

## 4. Benchmark Strategy

### 4.1 Minimal Benchmark Fixture

Add to `tests/test_demean.py`:

```python
import pytest
import numpy as np
from pyfixest.core.demean import demean
from pyfixest.core.demean_accelerated import demean_accelerated

@pytest.fixture
def benchmark_data_small():
    """Small dataset for quick iteration."""
    rng = np.random.default_rng(42)
    n, k = 10_000, 5
    return {
        'x': rng.normal(0, 1, (n, k)),
        'flist': np.column_stack([
            rng.integers(0, 100, n),
            rng.integers(0, 50, n),
        ]).astype(np.uint64),
        'weights': np.ones(n),
    }

@pytest.fixture
def benchmark_data_complex():
    """Complex FE structure from fixest benchmarks."""
    # Use generate_complex_fixed_effects_data() from test_demean.py
    X, flist, weights = generate_complex_fixed_effects_data()
    return {'x': X, 'flist': flist, 'weights': weights}

@pytest.mark.benchmark(group="demean")
def test_bench_demean_simple(benchmark, benchmark_data_small):
    data = benchmark_data_small
    result, success = benchmark(
        demean, data['x'], data['flist'], data['weights'], tol=1e-8
    )
    assert success

@pytest.mark.benchmark(group="demean")
def test_bench_demean_accelerated(benchmark, benchmark_data_small):
    data = benchmark_data_small
    result, success = benchmark(
        demean_accelerated, data['x'], data['flist'], data['weights'], tol=1e-8
    )
    assert success
```

### 4.2 Run Benchmarks

```bash
# Quick benchmark during iteration
pytest tests/test_demean.py -k "bench" --benchmark-only --benchmark-compare

# Full benchmark with stats
pytest tests/test_demean.py -k "bench" --benchmark-only \
    --benchmark-columns=mean,stddev,rounds \
    --benchmark-save=baseline
```

### 4.3 Benchmark Scenarios

| Scenario | n_samples | n_features | n_factors | n_groups_per_factor |
|----------|-----------|------------|-----------|---------------------|
| Small-simple | 10K | 5 | 2 | 100, 50 |
| Medium-2FE | 100K | 10 | 2 | 1000, 500 |
| Large-3FE | 1M | 5 | 3 | 5000, 2500, 100 |
| Complex | 100K | 3 | 3 | (per fixest) |

---

## 5. Implementation Roadmap

### Phase 1: Low-Hanging Fruit (Quick Wins)
1. [ ] Reduce per-column allocations (thread-local reuse)
2. [ ] Single contiguous buffer allocation
3. [ ] Add SIMD convergence check

### Phase 2: Algorithm Improvements
4. [ ] Implement grand acceleration
5. [ ] Add SSR-based convergence check
6. [ ] Specialized 2-FE path

### Phase 3: Advanced Optimization
7. [ ] Explicit SIMD for inner loops (wide crate)
8. [ ] Sort-by-group optimization
9. [ ] Coefficient-based iteration (major refactor)

---

## 6. Testing Requirements (Minimal)

Keep tests minimal for fast iteration:

```python
# Correctness: compare against pyhdfe (already in test_demean.py)
def test_accelerated_correctness():
    """Verify accelerated matches reference implementation."""
    X, flist, weights = generate_data()
    res_simple, _ = demean(X, flist, weights, tol=1e-10)
    res_accel, _ = demean_accelerated(X, flist, weights, tol=1e-10)
    assert np.allclose(res_simple, res_accel, rtol=1e-6, atol=1e-8)

# Benchmark: already covered above
```

---

## 7. Expected Performance Gains

| Optimization | Expected Gain | Effort |
|--------------|---------------|--------|
| Reduce allocations | 10-20% | Low |
| SIMD convergence | 5-10% | Low |
| Grand acceleration | 20-50% (hard problems) | Medium |
| 2-FE specialization | 10-30% (2-FE cases) | Medium |
| Full SIMD loops | 2-4x (compute-bound) | High |
| Coefficient-based | Variable | Very High |

**Realistic target**: 2-3x speedup over current `demean_accelerated.rs` for typical workloads, approaching fixest C++ performance.

---

## 8. Files to Modify

- `src/demean_accelerated.rs` - Main implementation
- `src/lib.rs` - Expose new functions if needed
- `pyfixest/core/demean_accelerated.py` - Python wrapper
- `tests/test_demean.py` - Add benchmarks
- `Cargo.toml` - Add `wide` crate for SIMD (optional)
