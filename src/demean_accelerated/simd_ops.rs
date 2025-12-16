//! SIMD-optimized convergence check for demeaning algorithms.
//!
//! Minimal subset used in production; other helpers were removed to avoid unused warnings.

use wide::{f64x4, CmpGe};

/// Check if maximum absolute difference is below tolerance with early exit.
#[inline]
pub fn is_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;
    let tol_vec = f64x4::splat(tol);

    // Process 4 elements at a time with early exit.
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let bv = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        let diff = (av - bv).abs();

        if diff.cmp_ge(tol_vec).any() {
            return false;
        }
    }

    // Handle remainder elements.
    for i in (chunks * 4)..n {
        if (a[i] - b[i]).abs() >= tol {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::is_converged;

    #[test]
    fn test_is_converged() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.001, 2.001, 3.001, 4.001];
        assert!(is_converged(&a, &b, 0.01));
        assert!(!is_converged(&a, &b, 0.0001));
    }
}
