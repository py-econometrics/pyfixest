//! Irons-Tuck acceleration for fixed-point iteration.
//!
//! This module provides the acceleration algorithm that speeds up convergence
//! of the alternating projections used in demeaning.

use crate::demean_accelerated::buffers::{indices, CoefficientBuffer};
use crate::demean_accelerated::simd_ops;

/// Trait for projection operations in fixed-point iteration.
///
/// A projector takes an input vector and projects it onto some subspace,
/// writing the result to the output buffer.
pub trait Projector {
    /// Project input onto the subspace, writing result to output.
    fn project(&mut self, input: &[f64], output: &mut [f64]);
}

/// Result of an acceleration step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepResult {
    /// Continue iteration
    Continue,
    /// Numerically converged (ssq == 0)
    NumericallyConverged,
}

/// Irons-Tuck acceleration for fixed-point iteration.
///
/// Uses a 3-point extrapolation scheme to accelerate convergence:
/// Given iterates X, G(X), G(G(X)), compute an optimal step that
/// minimizes the residual in the direction of the fixed point.
///
/// The algorithm:
/// 1. Compute G(X) and G(G(X))
/// 2. delta_GX = G(G(X)) - G(X)
/// 3. delta2_X = delta_GX - G(X) + X
/// 4. coef = (delta_GX · delta2_X) / (delta2_X · delta2_X)
/// 5. X_new = G(G(X)) - coef * delta_GX
pub struct IronsTuckAcceleration<P: Projector> {
    projector: P,
    buffers: CoefficientBuffer,
}

impl<P: Projector> IronsTuckAcceleration<P> {
    /// Create a new acceleration instance.
    ///
    /// # Arguments
    /// * `projector` - The projection operator (e.g., MultiFactorDemeaner)
    /// * `n_samples` - Number of samples (vector length)
    pub fn new(projector: P, n_samples: usize) -> Self {
        Self {
            projector,
            buffers: CoefficientBuffer::new(n_samples, indices::IRONS_TUCK_BUFFER_COUNT),
        }
    }

    /// Set the initial iterate.
    pub fn set_initial(&mut self, x: &[f64]) {
        self.buffers.buffer_mut(indices::X_CURR).copy_from_slice(x);
    }

    /// Get the current result.
    pub fn get_result(&self) -> &[f64] {
        self.buffers.buffer(indices::X_CURR)
    }

    /// Perform a single step of the iteration.
    ///
    /// # Arguments
    /// * `should_accelerate` - If true, apply Irons-Tuck acceleration;
    ///   otherwise, perform a regular projection step.
    ///
    /// # Returns
    /// `StepResult::NumericallyConverged` if the algorithm has converged
    /// numerically (ssq == 0), otherwise `StepResult::Continue`.
    pub fn step(&mut self, should_accelerate: bool) -> StepResult {
        // Store previous for convergence check
        self.buffers.copy_buffer(indices::X_CURR, indices::X_PREV);

        if should_accelerate {
            self.irons_tuck_step()
        } else {
            self.regular_step()
        }
    }

    /// Check if the iteration has converged.
    ///
    /// Uses SIMD-accelerated convergence check with early exit.
    pub fn is_converged(&self, tol: f64) -> bool {
        simd_ops::is_converged(
            self.buffers.buffer(indices::X_CURR),
            self.buffers.buffer(indices::X_PREV),
            tol,
        )
    }

    /// Regular projection step: X = G(X)
    pub fn regular_step(&mut self) -> StepResult {
        // Borrow buffers and projector as separate fields - no copy needed
        let (input, output) = self.buffers.get_read_write(indices::X_CURR, indices::GX_CURR);
        self.projector.project(input, output);

        // Copy result back: gx_curr -> x_curr
        self.buffers.copy_buffer(indices::GX_CURR, indices::X_CURR);

        StepResult::Continue
    }

    /// Irons-Tuck acceleration step.
    fn irons_tuck_step(&mut self) -> StepResult {
        let n = self.buffers.n_samples();

        // Compute G(X) -> gx_curr
        {
            let (input, output) = self.buffers.get_read_write(indices::X_CURR, indices::GX_CURR);
            self.projector.project(input, output);
        }

        // Compute G(G(X)) -> ggx_curr
        {
            let (input, output) = self.buffers.get_read_write(indices::GX_CURR, indices::GGX_CURR);
            self.projector.project(input, output);
        }

        // Apply acceleration
        self.apply_acceleration(n)
    }

    /// Apply Irons-Tuck acceleration formula.
    ///
    /// Uses loop unrolling for better performance. The compiler will auto-vectorize
    /// this on platforms with SIMD support.
    fn apply_acceleration(&mut self, n: usize) -> StepResult {
        let (data, stride) = self.buffers.raw_data_mut();

        let x_off = indices::X_CURR * stride;
        let gx_off = indices::GX_CURR * stride;
        let ggx_off = indices::GGX_CURR * stride;
        let dgx_off = indices::DELTA_GX * stride;
        let d2x_off = indices::DELTA2_X * stride;

        // Unrolled computation of delta_GX, delta2_X, vprod, ssq
        let chunks = n / 4;
        let mut vprod = 0.0;
        let mut ssq = 0.0;

        for i in 0..chunks {
            let idx = i * 4;

            // Load values
            let x0 = data[x_off + idx];
            let x1 = data[x_off + idx + 1];
            let x2 = data[x_off + idx + 2];
            let x3 = data[x_off + idx + 3];

            let gx0 = data[gx_off + idx];
            let gx1 = data[gx_off + idx + 1];
            let gx2 = data[gx_off + idx + 2];
            let gx3 = data[gx_off + idx + 3];

            let ggx0 = data[ggx_off + idx];
            let ggx1 = data[ggx_off + idx + 1];
            let ggx2 = data[ggx_off + idx + 2];
            let ggx3 = data[ggx_off + idx + 3];

            // Compute deltas
            let dgx0 = ggx0 - gx0;
            let dgx1 = ggx1 - gx1;
            let dgx2 = ggx2 - gx2;
            let dgx3 = ggx3 - gx3;

            let d2x0 = dgx0 - gx0 + x0;
            let d2x1 = dgx1 - gx1 + x1;
            let d2x2 = dgx2 - gx2 + x2;
            let d2x3 = dgx3 - gx3 + x3;

            // Store delta_gx and delta2_x
            data[dgx_off + idx] = dgx0;
            data[dgx_off + idx + 1] = dgx1;
            data[dgx_off + idx + 2] = dgx2;
            data[dgx_off + idx + 3] = dgx3;

            data[d2x_off + idx] = d2x0;
            data[d2x_off + idx + 1] = d2x1;
            data[d2x_off + idx + 2] = d2x2;
            data[d2x_off + idx + 3] = d2x3;

            // Accumulate dot products
            vprod += dgx0 * d2x0 + dgx1 * d2x1 + dgx2 * d2x2 + dgx3 * d2x3;
            ssq += d2x0 * d2x0 + d2x1 * d2x1 + d2x2 * d2x2 + d2x3 * d2x3;
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            let x = data[x_off + i];
            let gx = data[gx_off + i];
            let ggx = data[ggx_off + i];

            let delta_gx = ggx - gx;
            let delta2_x = delta_gx - gx + x;

            data[dgx_off + i] = delta_gx;
            data[d2x_off + i] = delta2_x;

            vprod += delta_gx * delta2_x;
            ssq += delta2_x * delta2_x;
        }

        if ssq == 0.0 {
            return StepResult::NumericallyConverged;
        }

        let coef = vprod / ssq;

        // Update: X = GGX - coef * delta_GX (unrolled)
        for i in 0..chunks {
            let idx = i * 4;
            data[x_off + idx] = data[ggx_off + idx] - coef * data[dgx_off + idx];
            data[x_off + idx + 1] = data[ggx_off + idx + 1] - coef * data[dgx_off + idx + 1];
            data[x_off + idx + 2] = data[ggx_off + idx + 2] - coef * data[dgx_off + idx + 2];
            data[x_off + idx + 3] = data[ggx_off + idx + 3] - coef * data[dgx_off + idx + 3];
        }

        // Handle remainder
        for i in (chunks * 4)..n {
            data[x_off + i] = data[ggx_off + i] - coef * data[dgx_off + i];
        }

        StepResult::Continue
    }
}

/// Grand acceleration for coarse-timescale convergence improvement.
///
/// While Irons-Tuck accelerates on every few iterations, Grand Acceleration
/// operates on a slower timescale, collecting 3 snapshots over many iterations
/// and then applying Irons-Tuck extrapolation to these coarse samples.
///
/// This helps with problems that have slow-decaying modes that fine-grained
/// acceleration doesn't capture well.
pub struct GrandAcceleration {
    /// History buffer: Y (snapshot 0), GY (snapshot 1), GGY (snapshot 2)
    history: CoefficientBuffer,
    /// Current position in the 3-point cycle (0, 1, or 2)
    counter: usize,
    /// How many iterations between snapshots
    interval: usize,
    /// Number of samples per vector
    n_samples: usize,
    /// True after we've completed recording all 3 points (counter just wrapped)
    ready_to_apply: bool,
}

/// Buffer indices for grand acceleration history
mod grand_indices {
    pub const Y: usize = 0;
    pub const GY: usize = 1;
    pub const GGY: usize = 2;
    pub const BUFFER_COUNT: usize = 3;
}

impl GrandAcceleration {
    /// Create a new GrandAcceleration.
    ///
    /// # Arguments
    /// * `n_samples` - Vector length
    /// * `interval` - Iterations between snapshots (default: 15)
    pub fn new(n_samples: usize, interval: usize) -> Self {
        Self {
            history: CoefficientBuffer::new(n_samples, grand_indices::BUFFER_COUNT),
            counter: 0,
            interval,
            n_samples,
            ready_to_apply: false,
        }
    }

    /// Check if this iteration should trigger a snapshot.
    #[inline]
    pub fn should_record(&self, iteration: usize) -> bool {
        iteration > 0 && iteration % self.interval == 0
    }

    /// Record the current iterate to the appropriate history slot.
    ///
    /// Call this when `should_record(iteration)` returns true.
    pub fn record(&mut self, x: &[f64]) {
        let slot = match self.counter {
            0 => grand_indices::Y,
            1 => grand_indices::GY,
            _ => grand_indices::GGY,
        };
        self.history.buffer_mut(slot).copy_from_slice(x);
        self.counter += 1;

        // Mark ready when we've completed recording all 3 points
        if self.counter == 3 {
            self.counter = 0;
            self.ready_to_apply = true;
        } else {
            self.ready_to_apply = false;
        }
    }

    /// Check if we have collected all 3 history points and can apply acceleration.
    ///
    /// Returns true immediately after recording the third point (GGY).
    #[inline]
    pub fn can_apply(&self) -> bool {
        self.ready_to_apply
    }

    /// Apply Irons-Tuck acceleration to the 3-point history.
    ///
    /// Modifies `x` in place with the accelerated value.
    /// Should only be called when `can_apply()` returns true.
    pub fn apply(&mut self, x: &mut [f64]) {
        let (data, stride) = self.history.raw_data_mut();

        let y_off = grand_indices::Y * stride;
        let gy_off = grand_indices::GY * stride;
        let ggy_off = grand_indices::GGY * stride;

        // Compute Irons-Tuck on the coarse history
        // delta_GY = GGY - GY
        // delta2_Y = delta_GY - GY + Y = GGY - 2*GY + Y
        // coef = (delta_GY · delta2_Y) / (delta2_Y · delta2_Y)
        // X_new = GGY - coef * delta_GY

        let mut vprod = 0.0;
        let mut ssq = 0.0;

        for i in 0..self.n_samples {
            let y = data[y_off + i];
            let gy = data[gy_off + i];
            let ggy = data[ggy_off + i];

            let delta_gy = ggy - gy;
            let delta2_y = delta_gy - gy + y;

            vprod += delta_gy * delta2_y;
            ssq += delta2_y * delta2_y;
        }

        // Only apply if we have meaningful curvature
        if ssq > 1e-30 {
            let coef = vprod / ssq;

            // Update x with accelerated value
            for i in 0..self.n_samples {
                let ggy = data[ggy_off + i];
                let gy = data[gy_off + i];
                let delta_gy = ggy - gy;
                x[i] = ggy - coef * delta_gy;
            }
        }
    }

    /// Reset the acceleration state (e.g., after a restart).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.counter = 0;
        self.ready_to_apply = false;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple identity projector for testing
    struct IdentityProjector;

    impl Projector for IdentityProjector {
        fn project(&mut self, input: &[f64], output: &mut [f64]) {
            output.copy_from_slice(input);
        }
    }

    /// Projector that scales by a constant
    struct ScaleProjector {
        scale: f64,
    }

    impl Projector for ScaleProjector {
        fn project(&mut self, input: &[f64], output: &mut [f64]) {
            for (out, &inp) in output.iter_mut().zip(input) {
                *out = inp * self.scale;
            }
        }
    }

    /// Projector that moves toward a target
    struct TowardTargetProjector {
        target: Vec<f64>,
        rate: f64,
    }

    impl Projector for TowardTargetProjector {
        fn project(&mut self, input: &[f64], output: &mut [f64]) {
            for (i, (out, &inp)) in output.iter_mut().zip(input).enumerate() {
                // Move rate fraction toward target
                *out = inp + self.rate * (self.target[i] - inp);
            }
        }
    }

    #[test]
    fn test_identity_converges_immediately() {
        let projector = IdentityProjector;
        let mut accel = IronsTuckAcceleration::new(projector, 4);

        let initial = vec![1.0, 2.0, 3.0, 4.0];
        accel.set_initial(&initial);

        // Single step should leave values unchanged
        accel.step(false);

        let result = accel.get_result();
        for (i, &v) in result.iter().enumerate() {
            assert!((v - initial[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convergence_check() {
        let projector = IdentityProjector;
        let mut accel = IronsTuckAcceleration::new(projector, 4);

        let initial = vec![1.0, 2.0, 3.0, 4.0];
        accel.set_initial(&initial);

        // After step, should be converged since identity doesn't change
        accel.step(false);

        assert!(accel.is_converged(1e-8));
    }

    #[test]
    fn test_scale_projector_converges_to_zero() {
        let projector = ScaleProjector { scale: 0.5 };
        let mut accel = IronsTuckAcceleration::new(projector, 4);

        let initial = vec![16.0, 32.0, 64.0, 128.0];
        accel.set_initial(&initial);

        // After 25 steps with scale 0.5: 128 * 0.5^25 ≈ 3.8e-6 < 1e-4
        for _ in 0..25 {
            accel.step(false);
        }

        let result = accel.get_result();
        for &v in result {
            assert!(v.abs() < 1e-4, "Expected < 1e-4, got {}", v);
        }
    }

    #[test]
    fn test_toward_target_with_acceleration() {
        let target = vec![10.0, 20.0, 30.0, 40.0];
        let projector = TowardTargetProjector {
            target: target.clone(),
            rate: 0.3,
        };
        let mut accel = IronsTuckAcceleration::new(projector, 4);

        let initial = vec![0.0, 0.0, 0.0, 0.0];
        accel.set_initial(&initial);

        // Run with acceleration every 3rd step
        for i in 0..30 {
            let should_accelerate = i % 3 == 0 && i > 0;
            accel.step(should_accelerate);
        }

        // Should be close to target
        let result = accel.get_result();
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - target[i]).abs() < 1.0,
                "result[{}] = {} should be close to {}",
                i,
                v,
                target[i]
            );
        }
    }

    #[test]
    fn test_acceleration_step_returns_step_result() {
        let projector = IdentityProjector;
        let mut accel = IronsTuckAcceleration::new(projector, 4);

        let initial = vec![1.0, 2.0, 3.0, 4.0];
        accel.set_initial(&initial);

        // With identity projector, acceleration should detect numerical convergence
        let result = accel.step(true);
        assert_eq!(result, StepResult::NumericallyConverged);
    }

    // --- GrandAcceleration tests ---

    #[test]
    fn test_grand_acceleration_should_record() {
        let grand = GrandAcceleration::new(4, 15);

        // Should not record at iteration 0
        assert!(!grand.should_record(0));

        // Should record at multiples of interval
        assert!(grand.should_record(15));
        assert!(grand.should_record(30));
        assert!(grand.should_record(45));

        // Should not record at non-multiples
        assert!(!grand.should_record(14));
        assert!(!grand.should_record(16));
    }

    #[test]
    fn test_grand_acceleration_counter_cycles() {
        let mut grand = GrandAcceleration::new(4, 15);

        let x = vec![1.0, 2.0, 3.0, 4.0];

        // Initially can't apply
        assert!(!grand.can_apply());

        // Record first point (Y)
        grand.record(&x);
        assert!(!grand.can_apply());

        // Record second point (GY)
        grand.record(&x);
        assert!(!grand.can_apply());

        // Record third point (GGY) - now can apply
        grand.record(&x);
        assert!(grand.can_apply());

        // Recording again starts new cycle, can_apply becomes false
        grand.record(&x);
        assert!(!grand.can_apply());

        // Complete another cycle
        grand.record(&x);
        grand.record(&x);
        assert!(grand.can_apply());
    }

    #[test]
    fn test_grand_acceleration_apply() {
        let mut grand = GrandAcceleration::new(4, 15);

        // Record converging sequence: Y, GY = Y/2, GGY = Y/4
        let y = vec![16.0, 32.0, 64.0, 128.0];
        let gy = vec![8.0, 16.0, 32.0, 64.0];
        let ggy = vec![4.0, 8.0, 16.0, 32.0];

        grand.record(&y);
        grand.record(&gy);
        grand.record(&ggy);

        assert!(grand.can_apply());

        // Apply should extrapolate toward the fixed point (0)
        let mut x = ggy.clone();
        grand.apply(&mut x);

        // With geometric convergence, Irons-Tuck should accelerate toward 0
        for &v in &x {
            assert!(v.abs() < 32.0, "Expected acceleration toward 0, got {}", v);
        }
    }

    #[test]
    fn test_grand_acceleration_reset() {
        let mut grand = GrandAcceleration::new(4, 15);

        let x = vec![1.0, 2.0, 3.0, 4.0];
        grand.record(&x);
        grand.record(&x);

        grand.reset();

        // After reset, counter should be 0 and can_apply false
        assert!(!grand.can_apply());

        // Need to record 3 new points
        grand.record(&x);
        assert!(!grand.can_apply());
    }
}
