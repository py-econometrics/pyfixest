//! Multi-FE (3+) solver with multi-phase strategy and projection operations.

use crate::demean_accelerated::buffers::{MultiFEBuffers, TwoFEBuffers};
use crate::demean_accelerated::types::{
    converged, irons_tuck_accelerate, should_continue, DemeanContext, FixestConfig,
};

use super::two_fe::run_2fe_acceleration;

// =============================================================================
// Projection Operations
// =============================================================================

/// Q-FE projection: Compute G(coef_in) -> coef_out.
///
/// Updates FEs in reverse order (Q-1 down to 0) matching fixest.
#[inline(always)]
fn project_qfe(
    ctx: &DemeanContext,
    in_out: &[f64],
    coef_in: &[f64],
    coef_out: &mut [f64],
    sum_other_means: &mut [f64],
) {
    let n_fe = ctx.index.n_fe;
    let n_obs = ctx.index.n_obs;

    let group_ids_ptr = ctx.index.group_ids.as_ptr();
    let coef_start = &ctx.index.coef_start;
    let sum_other_ptr = sum_other_means.as_mut_ptr();
    let coef_in_ptr = coef_in.as_ptr();
    let coef_out_ptr = coef_out.as_mut_ptr();
    let obs_weights_ptr = ctx.weights.per_obs.as_ptr();

    // Specialized fast path for 3 FEs (common case)
    if n_fe == 3 && ctx.weights.is_uniform {
        project_qfe_3fe_unweighted(
            n_obs,
            group_ids_ptr,
            coef_start,
            sum_other_ptr,
            coef_in_ptr,
            coef_out_ptr,
            in_out,
            &ctx.index.n_groups,
            &ctx.weights.per_group,
        );
        return;
    }

    // General case
    project_qfe_general(
        ctx,
        in_out,
        n_fe,
        n_obs,
        group_ids_ptr,
        coef_start,
        sum_other_ptr,
        coef_in_ptr,
        coef_out_ptr,
        obs_weights_ptr,
    );
}

/// Specialized 3-FE projection with loop unrolling.
///
/// Uses unsafe raw pointer arithmetic for performance - benchmarks show ~1.5x
/// speedup over safe iterator-based code for this hot path.
#[inline(always)]
fn project_qfe_3fe_unweighted(
    n_obs: usize,
    group_ids_ptr: *const usize,
    coef_start: &[usize],
    sum_other_ptr: *mut f64,
    coef_in_ptr: *const f64,
    coef_out_ptr: *mut f64,
    in_out: &[f64],
    n_groups: &[usize],
    group_weights: &[f64],
) {
    let (start_0, start_1, start_2) = (coef_start[0], coef_start[1], coef_start[2]);
    let fe_0_ptr = group_ids_ptr;
    let fe_1_ptr = unsafe { group_ids_ptr.add(n_obs) };
    let fe_2_ptr = unsafe { group_ids_ptr.add(2 * n_obs) };
    let in_out_ptr = in_out.as_ptr();

    let n_chunks = n_obs / 4;
    let remainder = n_obs % 4;

    // q=2: Process FE 2
    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let (g0_0, g0_1, g0_2, g0_3) = (
                *fe_0_ptr.add(base),
                *fe_0_ptr.add(base + 1),
                *fe_0_ptr.add(base + 2),
                *fe_0_ptr.add(base + 3),
            );
            let (g1_0, g1_1, g1_2, g1_3) = (
                *fe_1_ptr.add(base),
                *fe_1_ptr.add(base + 1),
                *fe_1_ptr.add(base + 2),
                *fe_1_ptr.add(base + 3),
            );
            *sum_other_ptr.add(base) =
                *coef_in_ptr.add(start_0 + g0_0) + *coef_in_ptr.add(start_1 + g1_0);
            *sum_other_ptr.add(base + 1) =
                *coef_in_ptr.add(start_0 + g0_1) + *coef_in_ptr.add(start_1 + g1_1);
            *sum_other_ptr.add(base + 2) =
                *coef_in_ptr.add(start_0 + g0_2) + *coef_in_ptr.add(start_1 + g1_2);
            *sum_other_ptr.add(base + 3) =
                *coef_in_ptr.add(start_0 + g0_3) + *coef_in_ptr.add(start_1 + g1_3);
        }
        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let (g0, g1) = (*fe_0_ptr.add(i), *fe_1_ptr.add(i));
            *sum_other_ptr.add(i) = *coef_in_ptr.add(start_0 + g0) + *coef_in_ptr.add(start_1 + g1);
        }
    }

    let n_groups_2 = n_groups[2];
    unsafe {
        std::ptr::copy_nonoverlapping(
            in_out_ptr.add(start_2),
            coef_out_ptr.add(start_2),
            n_groups_2,
        );
        for i in 0..n_obs {
            let g = *fe_2_ptr.add(i);
            *coef_out_ptr.add(start_2 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_2 {
            *coef_out_ptr.add(start_2 + g) /= *group_weights.get_unchecked(start_2 + g);
        }
    }

    // q=1: Process FE 1
    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let (g0_0, g0_1, g0_2, g0_3) = (
                *fe_0_ptr.add(base),
                *fe_0_ptr.add(base + 1),
                *fe_0_ptr.add(base + 2),
                *fe_0_ptr.add(base + 3),
            );
            let (g2_0, g2_1, g2_2, g2_3) = (
                *fe_2_ptr.add(base),
                *fe_2_ptr.add(base + 1),
                *fe_2_ptr.add(base + 2),
                *fe_2_ptr.add(base + 3),
            );
            *sum_other_ptr.add(base) =
                *coef_in_ptr.add(start_0 + g0_0) + *coef_out_ptr.add(start_2 + g2_0);
            *sum_other_ptr.add(base + 1) =
                *coef_in_ptr.add(start_0 + g0_1) + *coef_out_ptr.add(start_2 + g2_1);
            *sum_other_ptr.add(base + 2) =
                *coef_in_ptr.add(start_0 + g0_2) + *coef_out_ptr.add(start_2 + g2_2);
            *sum_other_ptr.add(base + 3) =
                *coef_in_ptr.add(start_0 + g0_3) + *coef_out_ptr.add(start_2 + g2_3);
        }
        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let (g0, g2) = (*fe_0_ptr.add(i), *fe_2_ptr.add(i));
            *sum_other_ptr.add(i) =
                *coef_in_ptr.add(start_0 + g0) + *coef_out_ptr.add(start_2 + g2);
        }
    }

    let n_groups_1 = n_groups[1];
    unsafe {
        std::ptr::copy_nonoverlapping(
            in_out_ptr.add(start_1),
            coef_out_ptr.add(start_1),
            n_groups_1,
        );
        for i in 0..n_obs {
            let g = *fe_1_ptr.add(i);
            *coef_out_ptr.add(start_1 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_1 {
            *coef_out_ptr.add(start_1 + g) /= *group_weights.get_unchecked(start_1 + g);
        }
    }

    // q=0: Process FE 0
    unsafe {
        for chunk in 0..n_chunks {
            let base = chunk * 4;
            let (g1_0, g1_1, g1_2, g1_3) = (
                *fe_1_ptr.add(base),
                *fe_1_ptr.add(base + 1),
                *fe_1_ptr.add(base + 2),
                *fe_1_ptr.add(base + 3),
            );
            let (g2_0, g2_1, g2_2, g2_3) = (
                *fe_2_ptr.add(base),
                *fe_2_ptr.add(base + 1),
                *fe_2_ptr.add(base + 2),
                *fe_2_ptr.add(base + 3),
            );
            *sum_other_ptr.add(base) =
                *coef_out_ptr.add(start_1 + g1_0) + *coef_out_ptr.add(start_2 + g2_0);
            *sum_other_ptr.add(base + 1) =
                *coef_out_ptr.add(start_1 + g1_1) + *coef_out_ptr.add(start_2 + g2_1);
            *sum_other_ptr.add(base + 2) =
                *coef_out_ptr.add(start_1 + g1_2) + *coef_out_ptr.add(start_2 + g2_2);
            *sum_other_ptr.add(base + 3) =
                *coef_out_ptr.add(start_1 + g1_3) + *coef_out_ptr.add(start_2 + g2_3);
        }
        for i in (n_chunks * 4)..(n_chunks * 4 + remainder) {
            let (g1, g2) = (*fe_1_ptr.add(i), *fe_2_ptr.add(i));
            *sum_other_ptr.add(i) =
                *coef_out_ptr.add(start_1 + g1) + *coef_out_ptr.add(start_2 + g2);
        }
    }

    let n_groups_0 = n_groups[0];
    unsafe {
        std::ptr::copy_nonoverlapping(
            in_out_ptr.add(start_0),
            coef_out_ptr.add(start_0),
            n_groups_0,
        );
        for i in 0..n_obs {
            let g = *fe_0_ptr.add(i);
            *coef_out_ptr.add(start_0 + g) -= *sum_other_ptr.add(i);
        }
        for g in 0..n_groups_0 {
            *coef_out_ptr.add(start_0 + g) /= *group_weights.get_unchecked(start_0 + g);
        }
    }
}

/// General Q-FE projection (any number of FEs).
///
/// Uses unsafe raw pointer arithmetic for performance.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn project_qfe_general(
    ctx: &DemeanContext,
    in_out: &[f64],
    n_fe: usize,
    n_obs: usize,
    group_ids_ptr: *const usize,
    coef_start: &[usize],
    sum_other_ptr: *mut f64,
    coef_in_ptr: *const f64,
    coef_out_ptr: *mut f64,
    obs_weights_ptr: *const f64,
) {
    let in_out_ptr = in_out.as_ptr();

    for q in (0..n_fe).rev() {
        unsafe {
            std::ptr::write_bytes(sum_other_ptr, 0, n_obs);
        }

        for h in 0..q {
            let start_h = coef_start[h];
            let fe_h_ptr = unsafe { group_ids_ptr.add(h * n_obs) };
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h_ptr.add(i);
                    *sum_other_ptr.add(i) += *coef_in_ptr.add(start_h + g);
                }
            }
        }

        for h in (q + 1)..n_fe {
            let start_h = coef_start[h];
            let fe_h_ptr = unsafe { group_ids_ptr.add(h * n_obs) };
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_h_ptr.add(i);
                    *sum_other_ptr.add(i) += *coef_out_ptr.add(start_h + g);
                }
            }
        }

        let start_q = coef_start[q];
        let n_groups_q = ctx.index.n_groups[q];
        let fe_q_ptr = unsafe { group_ids_ptr.add(q * n_obs) };
        let group_weights_q = ctx.weights.group_weights_for_fe(q, &ctx.index);

        unsafe {
            std::ptr::copy_nonoverlapping(
                in_out_ptr.add(start_q),
                coef_out_ptr.add(start_q),
                n_groups_q,
            );
        }

        if ctx.weights.is_uniform {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q_ptr.add(i);
                    *coef_out_ptr.add(start_q + g) -= *sum_other_ptr.add(i);
                }
            }
        } else {
            for i in 0..n_obs {
                unsafe {
                    let g = *fe_q_ptr.add(i);
                    *coef_out_ptr.add(start_q + g) -= *sum_other_ptr.add(i) * *obs_weights_ptr.add(i);
                }
            }
        }

        for g in 0..n_groups_q {
            unsafe {
                *coef_out_ptr.add(start_q + g) /= *group_weights_q.get_unchecked(g);
            }
        }
    }
}

// =============================================================================
// Solver
// =============================================================================

/// Solve 3+ FE demeaning using fixest's multi-phase strategy.
///
/// # Strategy
/// 1. Warmup iterations on all FEs
/// 2. 2-FE sub-convergence on first 2 FEs
/// 3. Re-acceleration on all FEs
pub fn solve_multi_fe(
    ctx: &DemeanContext,
    input: &[f64],
    config: &FixestConfig,
) -> (Vec<f64>, usize, bool) {
    let n_obs = ctx.index.n_obs;
    let n_coef = ctx.index.n_coef;
    let n0 = ctx.index.n_groups[0];
    let n1 = ctx.index.n_groups[1];
    let mut total_iter = 0usize;

    let mut mu = vec![0.0; n_obs];
    let mut coef = vec![0.0; n_coef];

    let mut multi_buffers = MultiFEBuffers::new(n_coef, n_obs);
    let mut two_buffers = TwoFEBuffers::new(n0, n1);

    // Phase 1: Warmup with all FEs (mu is zeros initially)
    let in_out_phase1 = ctx.scatter_to_coefficients(input);
    let (iter1, converged1) = run_qfe_acceleration(
        ctx,
        &in_out_phase1,
        &mut coef,
        &mut multi_buffers,
        config,
        config.iter_warmup,
        input,
    );
    total_iter += iter1;
    ctx.gather_and_add(&coef, &mut mu);

    if !converged1 {
        // Phase 2: 2-FE sub-convergence
        let in_out_phase2 = ctx.scatter_residuals_to_coefficients(input, &mu);
        let mut alpha = vec![0.0; n0];
        let mut beta = vec![0.0; n1];
        let in_out_2fe: Vec<f64> = in_out_phase2[..n0 + n1].to_vec();
        let effective_input: Vec<f64> = (0..n_obs).map(|i| input[i] - mu[i]).collect();

        let (iter2, _) = run_2fe_acceleration(
            ctx,
            &in_out_2fe,
            &mut alpha,
            &mut beta,
            &mut two_buffers,
            config,
            config.maxiter / 2,
            &effective_input,
        );
        total_iter += iter2;

        let fe0 = ctx.index.group_ids_for_fe(0);
        let fe1 = ctx.index.group_ids_for_fe(1);
        for i in 0..n_obs {
            mu[i] += alpha[fe0[i]] + beta[fe1[i]];
        }

        // Phase 3: Re-acceleration
        let remaining = config.maxiter.saturating_sub(total_iter);
        if remaining > 0 {
            let in_out_phase3 = ctx.scatter_residuals_to_coefficients(input, &mu);
            coef.fill(0.0);
            let (iter3, _) = run_qfe_acceleration(
                ctx,
                &in_out_phase3,
                &mut coef,
                &mut multi_buffers,
                config,
                remaining,
                input,
            );
            total_iter += iter3;
            ctx.gather_and_add(&coef, &mut mu);
        }
    }

    let mut output = vec![0.0; n_obs];
    for i in 0..n_obs {
        output[i] = input[i] - mu[i];
    }

    (output, total_iter, total_iter < config.maxiter)
}

/// Run Q-FE acceleration loop.
#[allow(clippy::too_many_arguments)]
fn run_qfe_acceleration(
    ctx: &DemeanContext,
    in_out: &[f64],
    coef: &mut [f64],
    buffers: &mut MultiFEBuffers,
    config: &FixestConfig,
    max_iter: usize,
    input: &[f64],
) -> (usize, bool) {
    let n_coef = ctx.index.n_coef;
    let n_fe = ctx.index.n_fe;
    let nb_coef_no_q = n_coef - ctx.index.n_groups[n_fe - 1];

    project_qfe(
        ctx,
        in_out,
        coef,
        &mut buffers.gx,
        &mut buffers.sum_other_means,
    );

    let mut keep_going = should_continue(&coef[..nb_coef_no_q], &buffers.gx[..nb_coef_no_q], config.tol);
    let mut iter = 0;
    let mut grand_counter = 0usize;
    let mut ssr = 0.0;

    while keep_going && iter < max_iter {
        iter += 1;

        project_qfe(
            ctx,
            in_out,
            &buffers.gx,
            &mut buffers.ggx,
            &mut buffers.sum_other_means,
        );

        if irons_tuck_accelerate(
            &mut coef[..nb_coef_no_q],
            &buffers.gx[..nb_coef_no_q],
            &buffers.ggx[..nb_coef_no_q],
        ) {
            break;
        }

        if iter >= config.iter_proj_after_acc {
            buffers.temp.copy_from_slice(coef);
            project_qfe(
                ctx,
                in_out,
                &buffers.temp,
                coef,
                &mut buffers.sum_other_means,
            );
        }

        project_qfe(
            ctx,
            in_out,
            coef,
            &mut buffers.gx,
            &mut buffers.sum_other_means,
        );
        keep_going = should_continue(&coef[..nb_coef_no_q], &buffers.gx[..nb_coef_no_q], config.tol);

        if iter % config.iter_grand_acc == 0 {
            grand_counter += 1;
            match grand_counter {
                1 => buffers.y[..nb_coef_no_q].copy_from_slice(&buffers.gx[..nb_coef_no_q]),
                2 => buffers.gy[..nb_coef_no_q].copy_from_slice(&buffers.gx[..nb_coef_no_q]),
                _ => {
                    buffers.ggy[..nb_coef_no_q].copy_from_slice(&buffers.gx[..nb_coef_no_q]);
                    if irons_tuck_accelerate(
                        &mut buffers.y[..nb_coef_no_q],
                        &buffers.gy[..nb_coef_no_q],
                        &buffers.ggy[..nb_coef_no_q],
                    ) {
                        break;
                    }
                    project_qfe(
                        ctx,
                        in_out,
                        &buffers.y,
                        &mut buffers.gx,
                        &mut buffers.sum_other_means,
                    );
                    grand_counter = 0;
                }
            }
        }

        if iter % 40 == 0 {
            let ssr_old = ssr;
            ctx.gather_and_subtract(&buffers.gx, input, &mut buffers.output_buf);
            ssr = buffers.output_buf.iter().map(|&r| r * r).sum();

            if iter > 40 && converged(ssr_old, ssr, config.tol) {
                keep_going = false;
                break;
            }
        }
    }

    coef.copy_from_slice(&buffers.gx);
    (iter, !keep_going)
}
