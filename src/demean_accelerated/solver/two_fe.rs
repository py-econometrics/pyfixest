//! 2-FE solver using the generic acceleration loop.

use crate::demean_accelerated::acceleration::{run_acceleration, DemeanBuffers};
use crate::demean_accelerated::projection::TwoFEProjector;
use crate::demean_accelerated::types::{DemeanContext, FixestConfig};

/// Solve 2-FE demeaning with acceleration.
pub fn solve_two_fe(
    ctx: &DemeanContext,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    let n_obs = ctx.index.n_obs;
    let n0 = ctx.index.n_groups[0];
    let n1 = ctx.index.n_groups[1];
    let n_coef = n0 + n1;

    // Scatter input to coefficient space
    let in_out = ctx.scatter_to_coefficients(input);

    // Initialize coefficient array (unified: [alpha | beta])
    let mut coef = vec![0.0; n_coef];

    // Create unified buffers
    let mut buffers = DemeanBuffers::new(n_coef, n_obs);

    // Run acceleration loop
    let (iter, converged) = run_acceleration::<TwoFEProjector>(
        ctx,
        &in_out,
        &mut coef,
        &mut buffers,
        config,
        config.maxiter,
        input,
    );

    // Reconstruct output: input - alpha - beta
    let mut result = vec![0.0; n_obs];
    let fe0 = ctx.index.group_ids_for_fe(0);
    let fe1 = ctx.index.group_ids_for_fe(1);

    for i in 0..n_obs {
        result[i] = input[i] - coef[fe0[i]] - coef[n0 + fe1[i]];
    }

    (result, iter, converged)
}

/// Run 2-FE acceleration loop (public for use by multi_fe solver).
#[allow(clippy::too_many_arguments)]
pub fn run_2fe_acceleration(
    ctx: &DemeanContext,
    in_out: &[f64],
    coef: &mut [f64],
    buffers: &mut DemeanBuffers,
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],
) -> (usize, bool) {
    run_acceleration::<TwoFEProjector>(ctx, in_out, coef, buffers, config, max_iter, input)
}
