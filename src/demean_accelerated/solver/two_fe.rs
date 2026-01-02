//! 2-FE accelerated solver with specialized projection operations.

use crate::demean_accelerated::types::{FEInfo, FixestConfig, irons_tuck_accelerate, should_continue, converged};
use crate::demean_accelerated::buffers::TwoFEBuffers;

// =============================================================================
// Projection Operations
// =============================================================================

/// 2-FE projection: Given alpha coefficients, compute new alpha via beta.
///
/// This matches fixest's compute_fe_coef_2 which avoids N-length intermediates
/// by working directly with coefficients.
#[inline(always)]
fn project_2fe(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha_in: &[f64],
    alpha_out: &mut [f64],
    beta: &mut [f64],
) {
    let n0 = fe_info.n_groups[0];
    let n1 = fe_info.n_groups[1];
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);
    let sw0 = fe_info.sum_weights_slice(0);
    let sw1 = fe_info.sum_weights_slice(1);
    let weights = &fe_info.weights;

    // Step 1: Compute beta from alpha_in
    beta[..n1].copy_from_slice(&in_out[n0..n0 + n1]);

    if fe_info.is_unweighted {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            beta[g1] -= alpha_in[g0];
        }
    } else {
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(weights.iter()) {
            beta[g1] -= alpha_in[g0] * w;
        }
    }

    beta.iter_mut().zip(sw1.iter()).for_each(|(b, sw)| *b /= sw);

    // Step 2: Compute alpha_out from beta
    alpha_out[..n0].copy_from_slice(&in_out[..n0]);

    if fe_info.is_unweighted {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            alpha_out[g0] -= beta[g1];
        }
    } else {
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(weights.iter()) {
            alpha_out[g0] -= beta[g1] * w;
        }
    }

    alpha_out.iter_mut().zip(sw0.iter()).for_each(|(a, sw)| *a /= sw);
}

/// Compute beta from alpha (half of project_2fe, for SSR computation).
#[inline(always)]
fn compute_beta_from_alpha(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha: &[f64],
    beta: &mut [f64],
) {
    let n1 = fe_info.n_groups[1];
    let n0 = fe_info.n_groups[0];
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);
    let sw1 = fe_info.sum_weights_slice(1);
    let weights = &fe_info.weights;

    beta[..n1].copy_from_slice(&in_out[n0..n0 + n1]);

    if fe_info.is_unweighted {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            beta[g1] -= alpha[g0];
        }
    } else {
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(weights.iter()) {
            beta[g1] -= alpha[g0] * w;
        }
    }

    beta.iter_mut().zip(sw1.iter()).for_each(|(b, sw)| *b /= sw);
}

// =============================================================================
// Solver
// =============================================================================

/// Solve 2-FE demeaning with acceleration.
pub fn solve_two_fe(
    fe_info: &FEInfo,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    let n_obs = fe_info.n_obs;
    let n0 = fe_info.n_groups[0];
    let n1 = fe_info.n_groups[1];

    let output = vec![0.0; n_obs];
    let in_out = fe_info.compute_in_out(input, &output);

    let mut alpha = vec![0.0; n0];
    let mut beta = vec![0.0; n1];
    let mut buffers = TwoFEBuffers::new(n0, n1);

    let (iter, converged) = run_2fe_acceleration(
        fe_info,
        &in_out,
        &mut alpha,
        &mut beta,
        &mut buffers,
        config,
        config.maxiter,
        input,
    );

    let mut result = vec![0.0; n_obs];
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);

    for i in 0..n_obs {
        result[i] = input[i] - alpha[fe0[i]] - beta[fe1[i]];
    }

    (result, iter, converged)
}

/// Run 2-FE acceleration loop with Irons-Tuck + Grand acceleration.
#[allow(clippy::too_many_arguments)]
pub fn run_2fe_acceleration(
    fe_info: &FEInfo,
    in_out: &[f64],
    alpha: &mut [f64],
    beta: &mut [f64],
    buffers: &mut TwoFEBuffers,
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],
) -> (usize, bool) {
    let n0 = fe_info.n_groups[0];
    let n_obs = fe_info.n_obs;

    let mut ssr = 0.0;
    let fe0 = fe_info.fe_ids_slice(0);
    let fe1 = fe_info.fe_ids_slice(1);

    project_2fe(fe_info, in_out, alpha, &mut buffers.gx, beta);

    let mut keep_going = should_continue(alpha, &buffers.gx, config.tol);
    let mut iter = 0;
    let mut grand_counter = 0usize;

    while keep_going && iter < max_iter {
        iter += 1;

        project_2fe(fe_info, in_out, &buffers.gx, &mut buffers.ggx, &mut buffers.beta_tmp);

        if irons_tuck_accelerate(alpha, &buffers.gx, &buffers.ggx) {
            break;
        }

        if iter >= config.iter_proj_after_acc {
            buffers.temp.copy_from_slice(alpha);
            project_2fe(fe_info, in_out, &buffers.temp, alpha, &mut buffers.beta_tmp);
        }

        project_2fe(fe_info, in_out, alpha, &mut buffers.gx, beta);

        keep_going = should_continue(alpha, &buffers.gx, config.tol);

        if iter % config.iter_grand_acc == 0 {
            grand_counter += 1;
            match grand_counter {
                1 => buffers.y.copy_from_slice(&buffers.gx),
                2 => buffers.gy.copy_from_slice(&buffers.gx),
                _ => {
                    buffers.ggy.copy_from_slice(&buffers.gx);
                    if irons_tuck_accelerate(&mut buffers.y, &buffers.gy, &buffers.ggy) {
                        break;
                    }
                    project_2fe(fe_info, in_out, &buffers.y, &mut buffers.gx, beta);
                    grand_counter = 0;
                }
            }
        }

        if iter % 40 == 0 {
            let ssr_old = ssr;
            compute_beta_from_alpha(fe_info, in_out, &buffers.gx, &mut buffers.beta_tmp);

            ssr = 0.0;
            for i in 0..n_obs {
                let resid = input[i] - buffers.gx[fe0[i]] - buffers.beta_tmp[fe1[i]];
                ssr += resid * resid;
            }

            if iter > 40 && converged(ssr_old, ssr, config.tol) {
                break;
            }
        }
    }

    alpha[..n0].copy_from_slice(&buffers.gx[..n0]);
    (iter, !keep_going)
}
