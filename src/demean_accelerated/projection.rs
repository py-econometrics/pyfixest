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
//!
//! # Usage with Accelerators
//!
//! Projectors are used with [`Accelerator`](crate::demean_accelerated::accelerator::Accelerator)
//! implementations that handle the iteration strategy (e.g., Irons-Tuck acceleration).

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
/// Projectors are used with [`Accelerator`](crate::demean_accelerated::accelerator::Accelerator)
/// implementations that handle the iteration strategy.
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

    // Step 1: Compute beta from alpha_in
    let beta = &mut scratch[..n1];
    compute_beta_from_alpha(ctx, in_out, &coef_in[..n0], beta);

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
/// matching fixest's algorithm.
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

    for q in (0..n_fe).rev() {
        // Zero the sum buffer
        sum_other_means.fill(0.0);

        // Sum coefficients from FEs before q (use coef_in)
        for h in 0..q {
            let start_h = ctx.index.coef_start[h];
            let fe_h = ctx.index.group_ids_for_fe(h);
            for (sum, &g) in sum_other_means.iter_mut().zip(fe_h.iter()) {
                *sum += coef_in[start_h + g];
            }
        }

        // Sum coefficients from FEs after q (use coef_out, already computed)
        for h in (q + 1)..n_fe {
            let start_h = ctx.index.coef_start[h];
            let fe_h = ctx.index.group_ids_for_fe(h);
            for (sum, &g) in sum_other_means.iter_mut().zip(fe_h.iter()) {
                *sum += coef_out[start_h + g];
            }
        }

        // Compute coef_out for FE q
        let start_q = ctx.index.coef_start[q];
        let n_groups_q = ctx.index.n_groups[q];
        let fe_q = ctx.index.group_ids_for_fe(q);
        let group_weights_q = ctx.weights.group_weights_for_fe(q, &ctx.index);

        // Copy in_out to coef_out for this FE
        coef_out[start_q..start_q + n_groups_q]
            .copy_from_slice(&in_out[start_q..start_q + n_groups_q]);

        // Subtract weighted sum_other_means
        if ctx.weights.is_uniform {
            for (&g, &sum) in fe_q.iter().zip(sum_other_means.iter()) {
                coef_out[start_q + g] -= sum;
            }
        } else {
            for ((&g, &sum), &w) in fe_q
                .iter()
                .zip(sum_other_means.iter())
                .zip(ctx.weights.per_obs.iter())
            {
                coef_out[start_q + g] -= sum * w;
            }
        }

        // Divide by group weights
        for (coef, &sw) in coef_out[start_q..start_q + n_groups_q]
            .iter_mut()
            .zip(group_weights_q.iter())
        {
            *coef /= sw;
        }
    }
}
