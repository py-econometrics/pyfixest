//! Multi-FE (3+) solver using the generic acceleration loop.

use crate::demean_accelerated::acceleration::{run_acceleration, DemeanBuffers};
use crate::demean_accelerated::projection::MultiFEProjector;
use crate::demean_accelerated::types::{DemeanContext, FixestConfig};

use super::two_fe::run_2fe_acceleration;

/// Solve 3+ FE demeaning using fixest's multi-phase strategy.
///
/// # Strategy
///
/// 1. **Warmup**: Run all-FE iterations to get initial estimates
/// 2. **2-FE sub-convergence**: Converge on first 2 FEs (faster)
/// 3. **Re-acceleration**: Final all-FE iterations to polish
pub fn solve_multi_fe(
    ctx: &DemeanContext,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    let n_obs = ctx.index.n_obs;
    let n_coef = ctx.index.n_coef;
    let n0 = ctx.index.n_groups[0];
    let n1 = ctx.index.n_groups[1];
    let n_coef_2fe = n0 + n1;
    let mut total_iter = 0usize;

    let mut mu = vec![0.0; n_obs];
    let mut coef = vec![0.0; n_coef];

    // Create buffers (one for multi-FE, one for 2-FE sub-convergence)
    let mut multi_buffers = DemeanBuffers::new(n_coef);
    let mut two_buffers = DemeanBuffers::new(n_coef_2fe);

    // Phase 1: Warmup with all FEs (mu is zeros initially)
    let in_out_phase1 = ctx.scatter_to_coefficients(input);
    let mut projector1 = MultiFEProjector::new(ctx, &in_out_phase1, input);
    let (iter1, converged1) = run_acceleration(
        &mut projector1,
        &mut coef,
        &mut multi_buffers,
        config,
        config.iter_warmup,
    );
    total_iter += iter1;
    ctx.gather_and_add(&coef, &mut mu);

    if !converged1 {
        // Phase 2: 2-FE sub-convergence
        let in_out_phase2 = ctx.scatter_residuals_to_coefficients(input, &mu);
        let mut coef_2fe = vec![0.0; n_coef_2fe];
        let in_out_2fe: Vec<f64> = in_out_phase2[..n_coef_2fe].to_vec();
        let effective_input: Vec<f64> = (0..n_obs).map(|i| input[i] - mu[i]).collect();

        let (iter2, _) = run_2fe_acceleration(
            ctx,
            &in_out_2fe,
            &mut coef_2fe,
            &mut two_buffers,
            config,
            config.maxiter / 2,
            &effective_input,
        );
        total_iter += iter2;

        // Add 2-FE coefficients to mu
        let fe0 = ctx.index.group_ids_for_fe(0);
        let fe1 = ctx.index.group_ids_for_fe(1);
        for i in 0..n_obs {
            mu[i] += coef_2fe[fe0[i]] + coef_2fe[n0 + fe1[i]];
        }

        // Phase 3: Re-acceleration with all FEs
        let remaining = config.maxiter.saturating_sub(total_iter);
        if remaining > 0 {
            let in_out_phase3 = ctx.scatter_residuals_to_coefficients(input, &mu);
            coef.fill(0.0);
            let mut projector3 = MultiFEProjector::new(ctx, &in_out_phase3, input);
            let (iter3, _) = run_acceleration(
                &mut projector3,
                &mut coef,
                &mut multi_buffers,
                config,
                remaining,
            );
            total_iter += iter3;
            ctx.gather_and_add(&coef, &mut mu);
        }
    }

    // Compute output: input - mu
    let mut output = vec![0.0; n_obs];
    for i in 0..n_obs {
        output[i] = input[i] - mu[i];
    }

    (output, total_iter, total_iter < config.maxiter)
}
