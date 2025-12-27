//! Acceleration strategies for fixed effects demeaning.
//!
//! This module provides the [`Accelerator`] trait for iteration acceleration,
//! with the default implementation [`IronsTuckGrand`] matching fixest's algorithm.

use crate::demean_accelerated::projection::Projector;
use crate::demean_accelerated::types::FixestConfig;

// =============================================================================
// Accelerator Trait
// =============================================================================

/// An acceleration strategy for iterative demeaning.
///
/// Accelerators take a [`Projector`] and repeatedly apply it until convergence,
/// using various techniques to speed up convergence.
///
/// # Associated Types
///
/// Each accelerator has its own buffer type, as different strategies require
/// different working memory (e.g., Irons-Tuck needs snapshots for extrapolation).
pub trait Accelerator {
    /// Working buffers needed by this acceleration strategy.
    type Buffers;

    /// Create buffers for the given coefficient count.
    fn create_buffers(n_coef: usize) -> Self::Buffers;

    /// Check if two scalar values have converged within tolerance.
    ///
    /// Uses both absolute and relative tolerance: converged if
    /// `|a - b| <= tol` OR `|a - b| <= tol * (0.1 + |a|)`.
    ///
    /// The `0.1` denominator offset prevents division by zero and provides
    /// a smooth transition between absolute tolerance (when |a| << 0.1) and
    /// relative tolerance (when |a| >> 0.1). This matches fixest's convergence check.
    ///
    /// # Implementation Note
    ///
    /// The relative tolerance check `|a - b| / (0.1 + |a|) <= tol` is rewritten
    /// as `|a - b| <= tol * (0.1 + |a|)` to avoid division, improving performance
    /// and SIMD-friendliness.
    #[inline]
    fn converged(a: f64, b: f64, tol: f64) -> bool {
        // 0.1 offset: ensures numerical stability and smooth absolute/relative transition
        const RELATIVE_TOL_OFFSET: f64 = 0.1;
        let diff = (a - b).abs();
        // Absolute tolerance check (faster, handles small values)
        // OR relative tolerance check (multiplication form, avoids division)
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

    /// Run the acceleration loop to convergence.
    ///
    /// # Arguments
    ///
    /// * `projector` - The projection operation to accelerate
    /// * `coef` - Initial coefficients (modified in place with final result)
    /// * `buffers` - Working buffers for the acceleration
    /// * `config` - Algorithm configuration (tolerance, etc.)
    /// * `max_iter` - Maximum iterations before giving up
    ///
    /// # Returns
    ///
    /// Tuple of (iterations_used, converged_flag)
    fn run<P: Projector>(
        projector: &mut P,
        coef: &mut [f64],
        buffers: &mut Self::Buffers,
        config: &FixestConfig,
        max_iter: usize,
    ) -> (usize, bool);
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
/// 2. **Grand acceleration**: Every `iter_grand_acc` iterations, applies Irons-Tuck
///    at a coarser level to accelerate long-range convergence.
///
/// Additionally, SSR (sum of squared residuals) is checked every 40 iterations
/// as a secondary convergence criterion. The interval of 40 balances overhead
/// (SSR computation is O(n)) against catching convergence that coefficient
/// checks might miss.
pub struct IronsTuckGrand;

/// Interval for SSR-based convergence checks (every N iterations).
/// Matches fixest's check frequency for secondary convergence criterion.
const SSR_CHECK_INTERVAL: usize = 40;

/// Buffers for Irons-Tuck + Grand acceleration.
pub struct IronsTuckGrandBuffers {
    /// G(x): Result of one projection step.
    pub gx: Vec<f64>,
    /// G(G(x)): Result of two projection steps.
    pub ggx: Vec<f64>,
    /// Temporary buffer for post-acceleration projection.
    pub temp: Vec<f64>,
    /// Grand acceleration: y snapshot.
    pub y: Vec<f64>,
    /// Grand acceleration: G(y) snapshot.
    pub gy: Vec<f64>,
    /// Grand acceleration: G(G(y)) snapshot.
    pub ggy: Vec<f64>,
}

impl IronsTuckGrand {
    /// Apply Irons-Tuck acceleration to speed up convergence.
    ///
    /// Given three successive iterates x, G(x), G(G(x)), computes an accelerated
    /// update that often converges faster than simple iteration.
    ///
    /// Returns `true` if already converged (denominator is zero), `false` otherwise.
    #[inline(always)]
    fn accelerate(x: &mut [f64], gx: &[f64], ggx: &[f64]) -> bool {
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
            return true;
        }

        let coef = vprod / ssq;
        x.iter_mut()
            .zip(gx.iter())
            .zip(ggx.iter())
            .for_each(|((x_i, &gx_i), &ggx_i)| {
                *x_i = ggx_i - coef * (ggx_i - gx_i);
            });

        false
    }
}

impl Accelerator for IronsTuckGrand {
    type Buffers = IronsTuckGrandBuffers;

    #[inline]
    fn create_buffers(n_coef: usize) -> Self::Buffers {
        IronsTuckGrandBuffers {
            gx: vec![0.0; n_coef],
            ggx: vec![0.0; n_coef],
            temp: vec![0.0; n_coef],
            y: vec![0.0; n_coef],
            gy: vec![0.0; n_coef],
            ggy: vec![0.0; n_coef],
        }
    }

    fn run<P: Projector>(
        projector: &mut P,
        coef: &mut [f64],
        buffers: &mut Self::Buffers,
        config: &FixestConfig,
        max_iter: usize,
    ) -> (usize, bool) {
        let conv_len = projector.convergence_len();

        // Initial projection
        projector.project(coef, &mut buffers.gx);

        let mut keep_going =
            Self::should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
        let mut iter = 0;
        let mut grand_counter = 0usize;
        let mut ssr = 0.0;

        while keep_going && iter < max_iter {
            iter += 1;

            // Double projection for Irons-Tuck: G(G(x))
            projector.project(&buffers.gx, &mut buffers.ggx);

            // Irons-Tuck acceleration
            if Self::accelerate(
                &mut coef[..conv_len],
                &buffers.gx[..conv_len],
                &buffers.ggx[..conv_len],
            ) {
                break;
            }

            // Post-acceleration projection (after warmup)
            if iter >= config.iter_proj_after_acc {
                buffers.temp[..conv_len].copy_from_slice(&coef[..conv_len]);
                projector.project(&buffers.temp, coef);
            }

            // Update gx for convergence check
            projector.project(coef, &mut buffers.gx);
            keep_going =
                Self::should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);

            // Grand acceleration (every iter_grand_acc iterations)
            if iter % config.iter_grand_acc == 0 {
                grand_counter += 1;
                match grand_counter {
                    1 => {
                        buffers.y[..conv_len].copy_from_slice(&buffers.gx[..conv_len]);
                    }
                    2 => {
                        buffers.gy[..conv_len].copy_from_slice(&buffers.gx[..conv_len]);
                    }
                    _ => {
                        buffers.ggy[..conv_len].copy_from_slice(&buffers.gx[..conv_len]);
                        if Self::accelerate(
                            &mut buffers.y[..conv_len],
                            &buffers.gy[..conv_len],
                            &buffers.ggy[..conv_len],
                        ) {
                            break;
                        }
                        projector.project(&buffers.y, &mut buffers.gx);
                        grand_counter = 0;
                    }
                }
            }

            // SSR convergence check (every SSR_CHECK_INTERVAL iterations)
            if iter % SSR_CHECK_INTERVAL == 0 {
                let ssr_old = ssr;
                ssr = projector.compute_ssr(&buffers.gx);

                if iter > SSR_CHECK_INTERVAL && Self::converged(ssr_old, ssr, config.tol) {
                    keep_going = false;
                    break;
                }
            }
        }

        // Copy final result
        coef.copy_from_slice(&buffers.gx);
        (iter, !keep_going)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demean_accelerated::projection::TwoFEProjector;
    use crate::demean_accelerated::types::DemeanContext;
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

        let n0 = ctx.index.n_groups[0];
        let n1 = ctx.index.n_groups[1];
        let n_coef = n0 + n1;

        let in_out = ctx.scatter_to_coefficients(&input);
        let mut coef = vec![0.0; n_coef];
        let mut buffers = IronsTuckGrand::create_buffers(n_coef);
        let mut projector = TwoFEProjector::new(&ctx, &in_out, &input);

        let (iter, converged) =
            IronsTuckGrand::run(&mut projector, &mut coef, &mut buffers, &config, config.maxiter);

        assert!(converged, "IronsTuckGrand should converge");
        assert!(iter < 100, "Should converge in less than 100 iterations");
    }
}
