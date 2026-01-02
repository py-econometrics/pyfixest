//! Generic acceleration loop for fixed effects demeaning.
//!
//! # Overview
//!
//! This module provides a generic implementation of the Irons-Tuck + Grand
//! acceleration loop used by both 2-FE and multi-FE solvers. By parameterizing
//! over the [`Projector`] trait, we avoid code duplication while maintaining
//! zero-cost abstraction through monomorphization.

use crate::demean_accelerated::projection::Projector;
use crate::demean_accelerated::types::{converged, irons_tuck_accelerate, should_continue, FixestConfig};

// =============================================================================
// Unified Buffer Struct
// =============================================================================

/// Working buffers for the acceleration loop.
///
/// Contains only the acceleration state vectors. Projection scratch
/// is owned by individual projectors.
pub struct DemeanBuffers {
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

impl DemeanBuffers {
    /// Create buffers for demeaning.
    ///
    /// # Arguments
    /// * `n_coef` - Total number of coefficients (sum of groups across all FEs)
    pub fn new(n_coef: usize) -> Self {
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
// Generic Acceleration Loop
// =============================================================================

/// Run the acceleration loop with any projector.
///
/// This is the core iteration loop used by both 2-FE and multi-FE solvers.
/// It implements Irons-Tuck acceleration with Grand acceleration every
/// `config.iter_grand_acc` iterations, and SSR convergence checking every
/// 40 iterations.
///
/// # Type Parameters
///
/// * `P` - The projector type (e.g., `TwoFEProjector` or `MultiFEProjector`)
///
/// # Returns
///
/// Tuple of (iterations_used, converged_flag)
pub fn run_acceleration<P: Projector>(
    projector: &mut P,
    in_out: &[f64],
    coef: &mut [f64],
    buffers: &mut DemeanBuffers,
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],
) -> (usize, bool) {
    let conv_len = projector.convergence_len();

    // Initial projection
    projector.project(in_out, coef, &mut buffers.gx);

    let mut keep_going = should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
    let mut iter = 0;
    let mut grand_counter = 0usize;
    let mut ssr = 0.0;

    while keep_going && iter < max_iter {
        iter += 1;

        // Double projection for Irons-Tuck: G(G(x))
        projector.project(in_out, &buffers.gx, &mut buffers.ggx);

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
            projector.project(in_out, &buffers.temp, coef);
        }

        // Update gx for convergence check
        projector.project(in_out, coef, &mut buffers.gx);
        keep_going = should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);

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
                    projector.project(in_out, &buffers.y, &mut buffers.gx);
                    grand_counter = 0;
                }
            }
        }

        // SSR convergence check (every 40 iterations)
        if iter % 40 == 0 {
            let ssr_old = ssr;
            ssr = projector.compute_ssr(in_out, &buffers.gx, input);

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
