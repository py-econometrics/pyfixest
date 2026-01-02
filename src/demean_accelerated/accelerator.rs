//! Acceleration strategies for fixed effects demeaning.
//!
//! This module provides the [`Accelerator`] trait for iteration acceleration,
//! with implementations for different strategies:
//!
//! - [`IronsTuckGrand`]: Irons-Tuck acceleration with Grand acceleration (default, matches fixest)
//! - [`SimpleIteration`]: Basic repeated projection for testing/debugging

use crate::demean_accelerated::projection::Projector;
use crate::demean_accelerated::types::{converged, irons_tuck_accelerate, should_continue, FixestConfig};

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
/// as a secondary convergence criterion.
pub struct IronsTuckGrand;

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
            should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
        let mut iter = 0;
        let mut grand_counter = 0usize;
        let mut ssr = 0.0;

        while keep_going && iter < max_iter {
            iter += 1;

            // Double projection for Irons-Tuck: G(G(x))
            projector.project(&buffers.gx, &mut buffers.ggx);

            // Irons-Tuck acceleration
            if irons_tuck_accelerate(
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
                should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);

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
                        if irons_tuck_accelerate(
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

            // SSR convergence check (every 40 iterations)
            if iter % 40 == 0 {
                let ssr_old = ssr;
                ssr = projector.compute_ssr(&buffers.gx);

                if iter > 40 && converged(ssr_old, ssr, config.tol) {
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

// =============================================================================
// SimpleIteration Accelerator
// =============================================================================

/// Simple repeated projection without acceleration.
///
/// This is a basic strategy that just applies G repeatedly until convergence:
/// `x_{n+1} = G(x_n)`. Useful for testing and debugging to verify that
/// projectors are correct, or as a baseline for comparing acceleration speedup.
///
/// **Note**: This converges much slower than [`IronsTuckGrand`] and is not
/// recommended for production use.
pub struct SimpleIteration;

/// Buffers for simple iteration (minimal).
pub struct SimpleIterationBuffers {
    /// G(x): Result of projection step.
    pub gx: Vec<f64>,
}

impl Accelerator for SimpleIteration {
    type Buffers = SimpleIterationBuffers;

    #[inline]
    fn create_buffers(n_coef: usize) -> Self::Buffers {
        SimpleIterationBuffers {
            gx: vec![0.0; n_coef],
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
            should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
        let mut iter = 0;

        while keep_going && iter < max_iter {
            iter += 1;

            // Simple iteration: x = G(x)
            coef[..conv_len].copy_from_slice(&buffers.gx[..conv_len]);
            projector.project(coef, &mut buffers.gx);

            keep_going =
                should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
        }

        // Copy final result
        coef.copy_from_slice(&buffers.gx);
        (iter, !keep_going)
    }
}
