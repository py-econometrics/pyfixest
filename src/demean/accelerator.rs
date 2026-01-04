//! Acceleration strategies for fixed effects demeaning.
//!
//! This module provides [`IronsTuckGrand`], the acceleration strategy matching
//! fixest's implementation.

use crate::demean::projection::Projector;
use crate::demean::types::{ConvergenceState, FixestConfig};

// =============================================================================
// Internal Types
// =============================================================================

/// Phase of grand acceleration state machine.
///
/// Grand acceleration applies Irons-Tuck at a coarser timescale to capture
/// long-range convergence patterns. It collects 3 snapshots of `gx` at
/// `iter_grand_acc` intervals, then applies Irons-Tuck to those snapshots.
///
/// # State transitions
///
/// ```text
/// Collect1st ──> Collect2nd ──> Collect3rdAndAccelerate ──┐
///     ^                                                   │
///     └───────────────────────────────────────────────────┘
/// ```
///
/// Actual acceleration happens every `3 × iter_grand_acc` iterations.
#[derive(Clone, Copy, Default)]
enum GrandPhase {
    /// Store current `gx` as first snapshot (y buffer).
    #[default]
    Collect1st,
    /// Store current `gx` as second snapshot (gy buffer).
    Collect2nd,
    /// Store current `gx` as third snapshot (ggy buffer), then accelerate.
    Collect3rdAndAccelerate,
}

/// Result of a grand acceleration step.
///
/// Grand acceleration operates on a coarser timescale than regular Irons-Tuck,
/// collecting snapshots every `iter_grand_acc` iterations to capture long-range
/// convergence patterns.
enum GrandStepResult {
    /// Continue with the next phase of the snapshot collection.
    Continue(GrandPhase),
    /// Grand acceleration detected convergence; iteration can stop.
    Done(ConvergenceState),
}

/// Buffers for Irons-Tuck with Grand acceleration.
///
/// # Regular Irons-Tuck buffers
///
/// - `gx`: G(x), result of one projection
/// - `ggx`: G(G(x)), result of two projections
/// - `temp`: temporary for post-acceleration projection
///
/// # Grand acceleration buffers
///
/// These store snapshots of `gx` at different times (separated by `iter_grand_acc`):
/// - `y`: 1st snapshot of gx
/// - `gy`: 2nd snapshot of gx
/// - `ggy`: 3rd snapshot of gx
///
/// Note: The names follow fixest's convention. Despite the names, these are NOT
/// nested projections (G(y), G(G(y))), but rather time-separated snapshots that
/// are then fed to Irons-Tuck as if they were successive iterates.
struct IronsTuckGrandBuffers {
    /// G(x): Result of one projection step (regular Irons-Tuck).
    gx: Vec<f64>,
    /// G(G(x)): Result of two projection steps (regular Irons-Tuck).
    ggx: Vec<f64>,
    /// Temporary buffer for post-acceleration projection.
    temp: Vec<f64>,
    /// Grand acceleration: 1st snapshot of gx.
    y: Vec<f64>,
    /// Grand acceleration: 2nd snapshot of gx.
    gy: Vec<f64>,
    /// Grand acceleration: 3rd snapshot of gx.
    ggy: Vec<f64>,
}

impl IronsTuckGrandBuffers {
    /// Create new buffers for the given coefficient count.
    fn new(n_coef: usize) -> Self {
        Self {
            gx: vec![0.0; n_coef],
            ggx: vec![0.0; n_coef],
            temp: vec![0.0; n_coef],
            y: vec![0.0; n_coef],
            gy: vec![0.0; n_coef],
            ggy: vec![0.0; n_coef],
        }
    }
}

// =============================================================================
// IronsTuckGrand Accelerator
// =============================================================================

/// Irons-Tuck acceleration with Grand acceleration.
///
/// This is the default acceleration strategy, matching fixest's implementation.
/// It combines two techniques:
///
/// 1. **Irons-Tuck**: After computing G(x) and G(G(x)), extrapolates to estimate
///    the fixed point directly using the formula from Irons & Tuck (1969).
///
/// 2. **Grand acceleration**: Every `iter_grand_acc` iteration applies Irons-Tuck
///    at a coarser level to accelerate long-range convergence.
///
/// Additionally, SSR (sum of squared residuals) is checked every `ssr_check_interval`
/// iteration as a secondary convergence criterion.
pub struct IronsTuckGrand {
    /// Algorithm configuration (tolerance, iteration parameters).
    config: FixestConfig,
    /// Working buffers for the acceleration algorithm.
    buffers: IronsTuckGrandBuffers,
}

impl IronsTuckGrand {
    /// Create a new accelerator with the given configuration and buffer size.
    #[inline]
    pub fn new(config: FixestConfig, n_coef: usize) -> Self {
        Self {
            config,
            buffers: IronsTuckGrandBuffers::new(n_coef),
        }
    }

    /// Run the acceleration loop to convergence.
    ///
    /// # Arguments
    ///
    /// * `projector` - The projection operation to accelerate
    /// * `coef` - Initial coefficients (modified in place with the final result)
    /// * `max_iter` - Maximum iterations before giving up
    ///
    /// # Returns
    ///
    /// Tuple of (iterations_used, convergence_state)
    pub fn run<P: Projector>(
        &mut self,
        projector: &mut P,
        coef: &mut [f64],
        max_iter: usize,
    ) -> (usize, ConvergenceState) {
        debug_assert_eq!(
            self.buffers.gx.len(),
            projector.coef_len(),
            "Accelerator buffer size ({}) must match projector coef_len ({})",
            self.buffers.gx.len(),
            projector.coef_len()
        );

        // Initial projection and convergence check
        let conv = self.project_and_check(projector, coef);
        if conv == ConvergenceState::Converged {
            return self.finalize_output(coef, 0, conv);
        }

        let mut grand_phase = GrandPhase::default();
        let mut ssr = 0.0;

        for iter in 1..=max_iter {
            // Core acceleration step
            let conv = self.acceleration_step_check(projector, coef, iter);
            if conv == ConvergenceState::Converged {
                return self.finalize_output(coef, iter, conv);
            }

            // Grand acceleration (every iter_grand_acc iterations)
            if iter % self.config.iter_grand_acc == 0 {
                let conv = self.grand_acceleration_check(projector, &mut grand_phase);
                if conv == ConvergenceState::Converged {
                    return self.finalize_output(coef, iter, conv);
                }
            }

            // SSR convergence check (every ssr_check_interval iterations)
            if iter % self.config.ssr_check_interval == 0 {
                let conv = self.ssr_convergence_check(projector, iter, &mut ssr);
                if conv == ConvergenceState::Converged {
                    return self.finalize_output(coef, iter, conv);
                }
            }
        }
        self.finalize_output(coef, max_iter, ConvergenceState::NotConverged)
    }

    /// Copy converged coefficients to the output buffer.
    ///
    /// This method should be called after `run()` has completed to retrieve
    /// the final coefficients from the internal `gx` buffer.
    #[inline]
    fn finalize_output(&self, coef: &mut [f64],
                           iter: usize,
                           convergence: ConvergenceState,) -> (usize, ConvergenceState) {
        coef.copy_from_slice(&self.buffers.gx);
        (iter, convergence)

    }

    /// Perform the core Irons-Tuck acceleration step.
    ///
    /// Returns `Converged` if convergence detected, `NotConverged` to continue.
    #[inline]
    fn acceleration_step_check<P: Projector>(
        &mut self,
        projector: &mut P,
        coef: &mut [f64],
        iter: usize,
    ) -> ConvergenceState {
        let conv_len = projector.convergence_len();

        // Double projection for Irons-Tuck: G(G(x))
        projector.project(&self.buffers.gx, &mut self.buffers.ggx);

        // Irons-Tuck acceleration
        if Self::accelerate(
            &mut coef[..conv_len],
            &self.buffers.gx[..conv_len],
            &self.buffers.ggx[..conv_len],
        ) == ConvergenceState::Converged
        {
            return ConvergenceState::Converged;
        }

        // Post-acceleration projection (after warmup)
        if iter >= self.config.iter_proj_after_acc {
            self.buffers.temp[..conv_len].copy_from_slice(&coef[..conv_len]);
            projector.project(&self.buffers.temp, coef);
        }

        // Update gx and check coefficient convergence
        self.project_and_check(projector, coef)
    }

    /// Perform grand acceleration and check for convergence.
    #[inline]
    fn grand_acceleration_check<P: Projector>(
        &mut self,
        projector: &mut P,
        grand_phase: &mut GrandPhase,
    ) -> ConvergenceState {
        match self.grand_acceleration_step(projector, *grand_phase) {
            GrandStepResult::Continue(next) => {
                *grand_phase = next;
                ConvergenceState::NotConverged
            }
            GrandStepResult::Done(state) => state,
        }
    }

    /// Check SSR-based convergence.
    #[inline]
    fn ssr_convergence_check<P: Projector>(
        &self,
        projector: &mut P,
        iter: usize,
        ssr: &mut f64,
    ) -> ConvergenceState {
        let ssr_old = *ssr;
        *ssr = projector.compute_ssr(&self.buffers.gx);

        if iter > self.config.ssr_check_interval && Self::converged(ssr_old, *ssr, self.config.tol)
        {
            ConvergenceState::Converged
        } else {
            ConvergenceState::NotConverged
        }
    }

    /// Project coefficients and check for convergence.
    #[inline]
    fn project_and_check<P: Projector>(
        &mut self,
        projector: &mut P,
        coef: &[f64],
    ) -> ConvergenceState {
        projector.project(coef, &mut self.buffers.gx);
        let conv_len = projector.convergence_len();
        if Self::should_continue(
            &coef[..conv_len],
            &self.buffers.gx[..conv_len],
            self.config.tol,
        ) {
            ConvergenceState::NotConverged
        } else {
            ConvergenceState::Converged
        }
    }

    /// Apply Irons-Tuck acceleration to speed up convergence.
    ///
    /// Given three successive iterates x, G(x), G(G(x)), extrapolates toward
    /// the fixed point using the formula from Irons & Tuck (1969).
    ///
    /// The method computes second differences `δ²x = G(G(x)) - 2G(x) + x` and uses
    /// them to estimate how far we are from the fixed point. If second differences
    /// are zero, we've already converged.
    #[inline(always)]
    fn accelerate(x: &mut [f64], gx: &[f64], ggx: &[f64]) -> ConvergenceState {
        let (vprod, ssq) = x
            .iter()
            .zip(gx.iter())
            .zip(ggx.iter())
            .map(|((&x_i, &gx_i), &ggx_i)| {
                let delta_gx = ggx_i - gx_i;
                let delta2_x = delta_gx - gx_i + x_i;
                (delta_gx * delta2_x, delta2_x * delta2_x)
            })
            .fold((0.0, 0.0), |(vp, sq), (dvp, dsq)| (vp + dvp, sq + dsq));

        if ssq == 0.0 {
            return ConvergenceState::Converged;
        }

        let coef = vprod / ssq;
        x.iter_mut()
            .zip(gx.iter())
            .zip(ggx.iter())
            .for_each(|((x_i, &gx_i), &ggx_i)| {
                *x_i = ggx_i - coef * (ggx_i - gx_i);
            });

        ConvergenceState::NotConverged
    }

    /// Perform one step of grand acceleration.
    ///
    /// Grand acceleration applies Irons-Tuck at a coarser timescale to capture
    /// long-range convergence patterns that fine-grained iteration might miss.
    ///
    /// # How it works
    ///
    /// Every `iter_grand_acc` iterations, this function is called to advance a
    /// 3-phase state machine:
    ///
    /// 1. **Collect1st**: Store current `gx` as the first snapshot (`y`)
    /// 2. **Collect2nd**: Store current `gx` as the second snapshot (`gy`)
    /// 3. **Collect3rdAndAccelerate**: Store current `gx` as third snapshot (`ggy`),
    ///    then apply Irons-Tuck to (y, gy, ggy) to extrapolate toward the fixed point
    ///
    /// After phase 3, the cycle repeats. This means actual acceleration happens
    /// every `3 × iter_grand_acc` iterations.
    #[inline]
    fn grand_acceleration_step<P: Projector>(
        &mut self,
        projector: &mut P,
        phase: GrandPhase,
    ) -> GrandStepResult {
        let conv_len = projector.convergence_len();
        match phase {
            GrandPhase::Collect1st => {
                self.buffers.y[..conv_len].copy_from_slice(&self.buffers.gx[..conv_len]);
                GrandStepResult::Continue(GrandPhase::Collect2nd)
            }
            GrandPhase::Collect2nd => {
                self.buffers.gy[..conv_len].copy_from_slice(&self.buffers.gx[..conv_len]);
                GrandStepResult::Continue(GrandPhase::Collect3rdAndAccelerate)
            }
            GrandPhase::Collect3rdAndAccelerate => {
                self.buffers.ggy[..conv_len].copy_from_slice(&self.buffers.gx[..conv_len]);
                let convergence = Self::accelerate(
                    &mut self.buffers.y[..conv_len],
                    &self.buffers.gy[..conv_len],
                    &self.buffers.ggy[..conv_len],
                );
                if convergence == ConvergenceState::Converged {
                    return GrandStepResult::Done(ConvergenceState::Converged);
                }
                projector.project(&self.buffers.y, &mut self.buffers.gx);
                GrandStepResult::Continue(GrandPhase::Collect1st)
            }
        }
    }

    /// Check if two scalar values have converged within tolerance.
    ///
    /// Uses both absolute and relative tolerance: converged if
    /// `|a - b| <= tol` OR `|a - b| <= tol * (0.1 + |a|)`.
    ///
    /// The `0.1` denominator offset prevents division by zero and provides
    /// a smooth transition between absolute tolerance (when |a| << 0.1) and
    /// relative tolerance (when |a| >> 0.1). This matches fixest's convergence check.
    #[inline]
    fn converged(a: f64, b: f64, tol: f64) -> bool {
        const RELATIVE_TOL_OFFSET: f64 = 0.1;
        let diff = (a - b).abs();
        (diff <= tol) || (diff <= tol * (RELATIVE_TOL_OFFSET + a.abs()))
    }

    /// Check if coefficient arrays have NOT converged (should keep iterating).
    ///
    /// Returns `true` if ANY pair of coefficients differs by more than tolerance.
    /// Uses early-exit: returns as soon as any non-converged pair is found.
    #[inline]
    fn should_continue(coef_old: &[f64], coef_new: &[f64], tol: f64) -> bool {
        coef_old
            .iter()
            .zip(coef_new.iter())
            .any(|(&a, &b)| !Self::converged(a, b, tol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demean::projection::TwoFEProjector;
    use crate::demean::types::DemeanContext;
    use ndarray::{Array1, Array2};

    /// Create a test problem with 2 fixed effects
    fn create_test_problem(n_obs: usize) -> (DemeanContext, Vec<f64>) {
        let n_fe = 2;
        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }
        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();
        (ctx, input)
    }

    #[test]
    fn test_irons_tuck_grand_convergence() {
        let (ctx, input) = create_test_problem(100);
        let config = FixestConfig::default();
        let maxiter = config.maxiter;

        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef = n0 + n1;

        let in_out = ctx.apply_design_matrix_t(&input);
        let mut coef = vec![0.0; n_coef];
        let mut accelerator = IronsTuckGrand::new(config, n_coef);
        let mut projector = TwoFEProjector::new(&ctx, &in_out, &input);

        let (iter, convergence) = accelerator.run(&mut projector, &mut coef, maxiter);

        assert_eq!(convergence, ConvergenceState::Converged, "IronsTuckGrand should converge");
        assert!(iter < 100, "Should converge in less than 100 iterations");
    }
}
