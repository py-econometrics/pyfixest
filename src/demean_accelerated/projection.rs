//! Projection operations for fixed effects demeaning.
//!
//! # Overview
//!
//! The demeaning algorithm iteratively applies a projection operator G that
//! updates coefficient estimates. Different FE counts have different projection
//! implementations, but they all share the same interface defined by [`Projector`].
//!
//! # Projection Semantics
//!
//! A projection takes current coefficient estimates and produces updated estimates:
//!
//! ```text
//! G: coef_in -> coef_out
//! ```
//!
//! The projection is defined such that repeated application converges to the
//! fixed effects solution: `G(G(G(...))) -> optimal coefficients`.

use crate::demean_accelerated::types::DemeanContext;

// =============================================================================
// Projector Trait
// =============================================================================

/// A projection operation for fixed-effects demeaning.
///
/// Projectors hold all context needed for projection: the [`DemeanContext`],
/// scattered input sums, original input values, and scratch buffers.
/// This makes the projection interface simple and clear.
///
/// # Performance
///
/// All methods are called in tight loops and should be marked `#[inline(always)]`.
/// Using static dispatch (`impl Projector` or generics) ensures zero overhead.
pub trait Projector {
    /// Project coefficients: coef_in → coef_out.
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]);

    /// Compute sum of squared residuals for the given coefficients.
    fn compute_ssr(&mut self, coef: &[f64]) -> f64;

    /// Length of coefficient slice to use for convergence checking.
    fn convergence_len(&self) -> usize;
}

// =============================================================================
// TwoFEProjector
// =============================================================================

/// Projector for 2 fixed effects.
///
/// Uses a specialized algorithm that works directly in coefficient space,
/// avoiding N-length intermediate arrays. This matches fixest's `compute_fe_coef_2`.
///
/// # Coefficient Layout
///
/// Coefficients are stored as `[alpha_0, ..., alpha_{n0-1}, beta_0, ..., beta_{n1-1}]`
/// where alpha are the coefficients for FE 0 and beta for FE 1.
pub struct TwoFEProjector<'a> {
    ctx: &'a DemeanContext,
    in_out: &'a [f64],
    input: &'a [f64],
    scratch: Vec<f64>,
}

impl<'a> TwoFEProjector<'a> {
    /// Create a new 2-FE projector.
    #[inline]
    pub fn new(ctx: &'a DemeanContext, in_out: &'a [f64], input: &'a [f64]) -> Self {
        let n1 = ctx.index.n_groups[1];
        Self {
            ctx,
            in_out,
            input,
            scratch: vec![0.0; n1],
        }
    }
}

impl Projector for TwoFEProjector<'_> {
    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        project_2fe(self.ctx, self.in_out, coef_in, coef_out, &mut self.scratch);
    }

    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        let ctx = self.ctx;
        let n0 = ctx.index.n_groups[0];
        let n_obs = ctx.index.n_obs;

        // Compute beta from alpha
        compute_beta_from_alpha(ctx, self.in_out, &coef[..n0], &mut self.scratch);

        // Compute SSR
        let fe0 = ctx.index.group_ids_for_fe(0);
        let fe1 = ctx.index.group_ids_for_fe(1);

        let mut ssr = 0.0;
        for i in 0..n_obs {
            let resid = self.input[i] - coef[fe0[i]] - self.scratch[fe1[i]];
            ssr += resid * resid;
        }
        ssr
    }

    #[inline(always)]
    fn convergence_len(&self) -> usize {
        self.ctx.index.n_groups[0]
    }
}

/// 2-FE projection: Given alpha coefficients, compute new alpha via beta.
#[inline(always)]
fn project_2fe(
    ctx: &DemeanContext,
    in_out: &[f64],
    coef_in: &[f64],
    coef_out: &mut [f64],
    scratch: &mut [f64],
) {
    let n0 = ctx.index.n_groups[0];
    let n1 = ctx.index.n_groups[1];
    let fe0 = ctx.index.group_ids_for_fe(0);
    let fe1 = ctx.index.group_ids_for_fe(1);
    let sw0 = ctx.weights.group_weights_for_fe(0, &ctx.index);
    let sw1 = ctx.weights.group_weights_for_fe(1, &ctx.index);

    // Use scratch[..n1] as beta
    let beta = &mut scratch[..n1];

    // Step 1: Compute beta from alpha_in (coef_in[..n0])
    beta.copy_from_slice(&in_out[n0..n0 + n1]);

    if ctx.weights.is_uniform {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            beta[g1] -= coef_in[g0];
        }
    } else {
        let obs_weights = &ctx.weights.per_obs;
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(obs_weights.iter()) {
            beta[g1] -= coef_in[g0] * w;
        }
    }

    beta.iter_mut()
        .zip(sw1.iter())
        .for_each(|(b, sw)| *b /= sw);

    // Step 2: Compute alpha_out from beta
    coef_out[..n0].copy_from_slice(&in_out[..n0]);

    if ctx.weights.is_uniform {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            coef_out[g0] -= beta[g1];
        }
    } else {
        let obs_weights = &ctx.weights.per_obs;
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(obs_weights.iter()) {
            coef_out[g0] -= beta[g1] * w;
        }
    }

    coef_out[..n0]
        .iter_mut()
        .zip(sw0.iter())
        .for_each(|(a, sw)| *a /= sw);

    // Copy beta to output
    coef_out[n0..n0 + n1].copy_from_slice(beta);
}

/// Compute beta from alpha (half of project_2fe, for SSR computation).
#[inline(always)]
fn compute_beta_from_alpha(ctx: &DemeanContext, in_out: &[f64], alpha: &[f64], beta: &mut [f64]) {
    let n0 = ctx.index.n_groups[0];
    let n1 = ctx.index.n_groups[1];
    let fe0 = ctx.index.group_ids_for_fe(0);
    let fe1 = ctx.index.group_ids_for_fe(1);
    let sw1 = ctx.weights.group_weights_for_fe(1, &ctx.index);

    beta[..n1].copy_from_slice(&in_out[n0..n0 + n1]);

    if ctx.weights.is_uniform {
        for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
            beta[g1] -= alpha[g0];
        }
    } else {
        let obs_weights = &ctx.weights.per_obs;
        for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(obs_weights.iter()) {
            beta[g1] -= alpha[g0] * w;
        }
    }

    beta.iter_mut()
        .zip(sw1.iter())
        .for_each(|(b, sw)| *b /= sw);
}

// =============================================================================
// MultiFEProjector
// =============================================================================

/// Projector for 3+ fixed effects.
///
/// Uses a general Q-FE projection that processes FEs in reverse order,
/// matching fixest's algorithm. Includes a specialized fast path for
/// 3-FE unweighted case.
pub struct MultiFEProjector<'a> {
    ctx: &'a DemeanContext,
    in_out: &'a [f64],
    input: &'a [f64],
    scratch: Vec<f64>,
}

impl<'a> MultiFEProjector<'a> {
    /// Create a new multi-FE projector.
    #[inline]
    pub fn new(ctx: &'a DemeanContext, in_out: &'a [f64], input: &'a [f64]) -> Self {
        let n_obs = ctx.index.n_obs;
        Self {
            ctx,
            in_out,
            input,
            scratch: vec![0.0; n_obs],
        }
    }
}

impl Projector for MultiFEProjector<'_> {
    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        project_qfe(self.ctx, self.in_out, coef_in, coef_out, &mut self.scratch);
    }

    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        let ctx = self.ctx;
        let n_obs = ctx.index.n_obs;
        let n_fe = ctx.index.n_fe;
        let mut ssr = 0.0;

        for i in 0..n_obs {
            let mut sum = 0.0;
            for q in 0..n_fe {
                let offset = ctx.index.coef_start[q];
                let g = ctx.index.group_ids[q * n_obs + i];
                sum += coef[offset + g];
            }
            let resid = self.input[i] - sum;
            ssr += resid * resid;
        }
        ssr
    }

    #[inline(always)]
    fn convergence_len(&self) -> usize {
        let ctx = self.ctx;
        ctx.index.n_coef - ctx.index.n_groups[ctx.index.n_fe - 1]
    }
}

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
            *sum_other_ptr.add(i) =
                *coef_in_ptr.add(start_0 + g0) + *coef_in_ptr.add(start_1 + g1);
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
                    *coef_out_ptr.add(start_q + g) -=
                        *sum_other_ptr.add(i) * *obs_weights_ptr.add(i);
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
