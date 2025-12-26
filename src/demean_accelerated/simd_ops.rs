//! Convergence check for demeaning algorithms.

/// Check if maximum absolute difference is below tolerance with early exit.
///
/// Uses loop unrolling for better performance. The compiler will auto-vectorize
/// this on platforms with SIMD support.
#[inline]
pub fn is_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    // Process 4 elements at a time with early exit (unrolled for auto-vectorization).
    for i in 0..chunks {
        let idx = i * 4;
        let d0 = (a[idx] - b[idx]).abs();
        let d1 = (a[idx + 1] - b[idx + 1]).abs();
        let d2 = (a[idx + 2] - b[idx + 2]).abs();
        let d3 = (a[idx + 3] - b[idx + 3]).abs();

        if d0 >= tol || d1 >= tol || d2 >= tol || d3 >= tol {
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
