//! Generic acceleration loop for fixed effects demeaning.
//!
//! # Overview
//!
//! This module provides a generic implementation of the Irons-Tuck + Grand
//! acceleration loop used by both 2-FE and multi-FE solvers. By parameterizing
//! over the [`Projector`] trait, we avoid code duplication while maintaining
//! zero-cost abstraction through monomorphization.
//!
//! # Algorithm
//!
//! The acceleration loop implements:
//!
//! 1. **Basic iteration**: x_{n+1} = G(x_n) where G is the projection operator
//! 2. **Irons-Tuck acceleration**: Computes optimal step size along acceleration direction
//! 3. **Grand acceleration**: Additional acceleration applied every few iterations
//! 4. **SSR convergence**: Checks sum of squared residuals every 40 iterations

use crate::demean_accelerated::projection::Projector;
use crate::demean_accelerated::types::{
    converged, irons_tuck_accelerate, should_continue, DemeanContext, FixestConfig,
};

// =============================================================================
// Acceleration Buffers
// =============================================================================

/// Shared buffers for the acceleration loop.
///
/// These buffers are used by all projector types and hold intermediate
/// results during the Irons-Tuck and Grand acceleration steps.
pub struct AccelBuffers {
    /// G(x): Result of one projection step.
    pub gx: Vec<f64>,
    /// G(G(x)): Result of two projection steps.
    pub ggx: Vec<f64>,
    /// Temporary buffer for post-acceleration projection.
    pub temp: Vec<f64>,

    // Grand acceleration buffers
    /// y: First snapshot for grand acceleration.
    pub y: Vec<f64>,
    /// G(y): Second snapshot for grand acceleration.
    pub gy: Vec<f64>,
    /// G(G(y)): Third snapshot for grand acceleration.
    pub ggy: Vec<f64>,
}

impl AccelBuffers {
    /// Create acceleration buffers for the given coefficient count.
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
/// # Arguments
///
/// * `ctx` - Demeaning context (index + weights)
/// * `in_out` - Scattered input sums in coefficient space
/// * `coef` - Coefficient buffer (in/out)
/// * `buffers` - Shared acceleration buffers
/// * `scratch` - Projector-specific scratch buffers
/// * `config` - Algorithm configuration
/// * `max_iter` - Maximum iterations for this run
/// * `input` - Original input values (for SSR computation)
///
/// # Returns
///
/// Tuple of (iterations_used, converged_flag)
///
/// # Performance
///
/// This function is monomorphized for each projector type, ensuring that
/// all `P::project` calls are inlined and optimized for that specific case.
#[allow(clippy::too_many_arguments)]
pub fn run_acceleration<P: Projector>(
    ctx: &DemeanContext,
    in_out: &[f64],
    coef: &mut [f64],
    buffers: &mut AccelBuffers,
    scratch: &mut P::Scratch,
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],
) -> (usize, bool) {
    let conv_len = P::convergence_len(ctx);

    // Initial projection
    P::project(ctx, in_out, coef, buffers.gx.as_mut_slice(), scratch);

    let mut keep_going = should_continue(&coef[..conv_len], &buffers.gx[..conv_len], config.tol);
    let mut iter = 0;
    let mut grand_counter = 0usize;
    let mut ssr = 0.0;

    while keep_going && iter < max_iter {
        iter += 1;

        // Double projection for Irons-Tuck: G(G(x))
        P::project(
            ctx,
            in_out,
            buffers.gx.as_slice(),
            buffers.ggx.as_mut_slice(),
            scratch,
        );

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
            P::project(
                ctx,
                in_out,
                buffers.temp.as_slice(),
                coef,
                scratch,
            );
        }

        // Update gx for convergence check
        P::project(ctx, in_out, coef, buffers.gx.as_mut_slice(), scratch);
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
                    P::project(
                        ctx,
                        in_out,
                        buffers.y.as_slice(),
                        buffers.gx.as_mut_slice(),
                        scratch,
                    );
                    grand_counter = 0;
                }
            }
        }

        // SSR convergence check (every 40 iterations)
        if iter % 40 == 0 {
            let ssr_old = ssr;
            ssr = P::compute_ssr(ctx, in_out, &buffers.gx, input, scratch);

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
