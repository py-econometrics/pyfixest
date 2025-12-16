//! SIMD-optimized operations for demeaning algorithms.
//!
//! Uses wide::f64x4 for portable SIMD (AVX on x86, NEON on ARM).
//! All functions handle non-aligned array lengths with scalar fallback.

use wide::{f64x4, CmpGe};

/// Compute maximum absolute difference between two slices.
/// Used for convergence checking with early exit.
///
/// Returns max |a[i] - b[i]| for all i.
#[inline]
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    let mut max_vec = f64x4::ZERO;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let bv = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        let diff = (av - bv).abs();
        max_vec = max_vec.max(diff);
    }

    // Reduce vector to scalar
    let arr = max_vec.to_array();
    let mut max_scalar = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);

    // Handle remainder elements
    for i in (chunks * 4)..n {
        max_scalar = max_scalar.max((a[i] - b[i]).abs());
    }

    max_scalar
}

/// Check if maximum absolute difference is below tolerance.
/// Early exit version - returns false as soon as any element exceeds tolerance.
#[inline]
pub fn is_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;
    let tol_vec = f64x4::splat(tol);

    // Process 4 elements at a time with early exit
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let bv = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        let diff = (av - bv).abs();

        // Check if any element exceeds tolerance
        let exceeds = diff.cmp_ge(tol_vec);
        if exceeds.any() {
            return false;
        }
    }

    // Handle remainder elements
    for i in (chunks * 4)..n {
        if (a[i] - b[i]).abs() >= tol {
            return false;
        }
    }

    true
}

/// Compute dot product of two slices.
/// Used for vprod calculation in Irons-Tuck acceleration.
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;

    let mut sum_vec = f64x4::ZERO;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let bv = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        sum_vec = sum_vec + (av * bv);
    }

    // Reduce vector to scalar
    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder elements
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }

    sum
}

/// Compute sum of squares of a slice.
/// Used for ssq calculation in Irons-Tuck acceleration.
#[inline]
pub fn sum_of_squares(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;

    let mut sum_vec = f64x4::ZERO;

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        sum_vec = sum_vec + (av * av);
    }

    // Reduce vector to scalar
    let arr = sum_vec.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];

    // Handle remainder elements
    for i in (chunks * 4)..n {
        sum += a[i] * a[i];
    }

    sum
}

/// Compute y = y + alpha * x (AXPY operation).
/// Used for coefficient vector updates.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len();
    let chunks = n / 4;
    let alpha_vec = f64x4::splat(alpha);

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let xv = f64x4::new([x[idx], x[idx + 1], x[idx + 2], x[idx + 3]]);
        let yv = f64x4::new([y[idx], y[idx + 1], y[idx + 2], y[idx + 3]]);
        let result = yv + alpha_vec * xv;
        let arr = result.to_array();
        y[idx] = arr[0];
        y[idx + 1] = arr[1];
        y[idx + 2] = arr[2];
        y[idx + 3] = arr[3];
    }

    // Handle remainder elements
    for i in (chunks * 4)..n {
        y[i] += alpha * x[i];
    }
}

/// Compute a = a - scale * b (scale and subtract).
/// Used for Irons-Tuck coefficient update: X = GGX - coef * delta_GX.
#[inline]
pub fn scale_sub(a: &mut [f64], b: &[f64], scale: f64) {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let chunks = n / 4;
    let scale_vec = f64x4::splat(scale);

    // Process 4 elements at a time
    for i in 0..chunks {
        let idx = i * 4;
        let av = f64x4::new([a[idx], a[idx + 1], a[idx + 2], a[idx + 3]]);
        let bv = f64x4::new([b[idx], b[idx + 1], b[idx + 2], b[idx + 3]]);
        let result = av - scale_vec * bv;
        let arr = result.to_array();
        a[idx] = arr[0];
        a[idx + 1] = arr[1];
        a[idx + 2] = arr[2];
        a[idx + 3] = arr[3];
    }

    // Handle remainder elements
    for i in (chunks * 4)..n {
        a[i] -= scale * b[i];
    }
}

/// Copy src to dst with SIMD acceleration.
#[inline]
pub fn copy_slice(src: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_abs_diff() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1, 2.0, 2.5, 4.0, 5.2];
        let result = max_abs_diff(&a, &b);
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_is_converged() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.001, 2.001, 3.001, 4.001];
        assert!(is_converged(&a, &b, 0.01));
        assert!(!is_converged(&a, &b, 0.0001));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        // 2 + 6 + 12 + 20 + 30 = 70
        let result = dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_sum_of_squares() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // 1 + 4 + 9 + 16 + 25 = 55
        let result = sum_of_squares(&a);
        assert!((result - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_axpy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0, 60.0]);
    }

    #[test]
    fn test_scale_sub() {
        let mut a = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        scale_sub(&mut a, &b, 2.0);
        assert_eq!(a, vec![8.0, 16.0, 24.0, 32.0, 40.0]);
    }

    #[test]
    fn test_empty_slices() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(max_abs_diff(&a, &b), 0.0);
        assert!(is_converged(&a, &b, 1e-8));
        assert_eq!(dot_product(&a, &b), 0.0);
        assert_eq!(sum_of_squares(&a), 0.0);
    }
}
