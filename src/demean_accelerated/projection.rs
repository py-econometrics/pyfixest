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
//! Projectors are used with [`IronsTuckGrand`](crate::demean_accelerated::accelerator::IronsTuckGrand)
//! which handles the iteration strategy.

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
/// Projectors are used with [`IronsTuckGrand`](crate::demean_accelerated::accelerator::IronsTuckGrand)
/// which handles the iteration strategy.
///
/// # Performance
///
/// All methods are called in tight loops and should be marked `#[inline(always)]`.
/// Using static dispatch (`impl Projector` or generics) ensures zero overhead.
pub trait Projector {
    /// Total number of coefficients this projector operates on.
    ///
    /// This defines the required size of coefficient arrays passed to
    /// `project()` and `compute_ssr()`. Accelerator buffers must be
    /// sized to match this value.
    fn coef_len(&self) -> usize;

    /// Project coefficients: coef_in → coef_out.
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]);

    /// Compute sum of squared residuals for the given coefficients.
    fn compute_ssr(&mut self, coef: &[f64]) -> f64;

    /// Length of coefficient slice to use for convergence checking.
    ///
    /// This may be smaller than `coef_len()` when not all coefficients
    /// need to be checked (e.g., for 2-FE only alpha is checked).
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

    /// Compute beta coefficients from alpha, storing result in scratch buffer.
    ///
    /// For each group g1 in FE1:
    ///   beta[g1] = (in_out[g1] - Σ alpha[g0] * w) / group_weight[g1]
    #[inline]
    fn compute_beta_from_alpha(&mut self, alpha: &[f64]) {
        let n0 = self.ctx.index.n_groups[0];
        let n1 = self.ctx.index.n_groups[1];
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);
        let sw1 = self.ctx.group_weights_for_fe(1);

        self.scratch[..n1].copy_from_slice(&self.in_out[n0..n0 + n1]);

        if self.ctx.weights.is_uniform {
            for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
                self.scratch[g1] -= alpha[g0];
            }
        } else {
            for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(self.ctx.weights.per_obs.iter())
            {
                self.scratch[g1] -= alpha[g0] * w;
            }
        }

        for (b, &sw) in self.scratch[..n1].iter_mut().zip(sw1.iter()) {
            *b /= sw;
        }
    }

    /// Compute alpha coefficients from beta (stored in scratch), writing to alpha_out.
    ///
    /// For each group g0 in FE0:
    ///   alpha[g0] = (in_out[g0] - Σ beta[g1] * w) / group_weight[g0]
    #[inline]
    fn compute_alpha_from_beta(&self, alpha_out: &mut [f64]) {
        let n0 = self.ctx.index.n_groups[0];
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);
        let sw0 = self.ctx.group_weights_for_fe(0);

        alpha_out[..n0].copy_from_slice(&self.in_out[..n0]);

        if self.ctx.weights.is_uniform {
            for (&g0, &g1) in fe0.iter().zip(fe1.iter()) {
                alpha_out[g0] -= self.scratch[g1];
            }
        } else {
            for ((&g0, &g1), &w) in fe0.iter().zip(fe1.iter()).zip(self.ctx.weights.per_obs.iter())
            {
                alpha_out[g0] -= self.scratch[g1] * w;
            }
        }

        for (a, &sw) in alpha_out[..n0].iter_mut().zip(sw0.iter()) {
            *a /= sw;
        }
    }
}

impl Projector for TwoFEProjector<'_> {
    #[inline(always)]
    fn coef_len(&self) -> usize {
        self.ctx.index.n_groups[0] + self.ctx.index.n_groups[1]
    }

    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        let n0 = self.ctx.index.n_groups[0];
        let n1 = self.ctx.index.n_groups[1];

        // Step 1: alpha_in -> beta
        self.compute_beta_from_alpha(&coef_in[..n0]);

        // Step 2: beta -> alpha_out
        self.compute_alpha_from_beta(coef_out);

        // Step 3: Copy beta to output
        coef_out[n0..n0 + n1].copy_from_slice(&self.scratch[..n1]);
    }

    /// Compute sum of squared residuals for the given coefficients.
    ///
    /// # Side Effects
    ///
    /// This method recomputes beta from alpha and stores it in `self.scratch`.
    /// After this call, `self.scratch[..n1]` contains the beta coefficients
    /// derived from `coef[..n0]` (the alpha coefficients).
    ///
    /// This is intentional: the SSR computation needs consistent alpha/beta pairs,
    /// and recomputing beta ensures correctness even if the caller's `coef` array
    /// has stale beta values.
    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        let n0 = self.ctx.index.n_groups[0];
        let fe0 = self.ctx.index.group_ids_for_fe(0);
        let fe1 = self.ctx.index.group_ids_for_fe(1);

        // Compute beta from alpha (updates self.scratch)
        self.compute_beta_from_alpha(&coef[..n0]);

        // Compute SSR: Σ (input[i] - alpha[fe0[i]] - beta[fe1[i]])²
        let mut ssr = 0.0;
        for ((&g0, &g1), &x) in fe0.iter().zip(fe1.iter()).zip(self.input.iter()) {
            let resid = x - coef[g0] - self.scratch[g1];
            ssr += resid * resid;
        }
        ssr
    }

    #[inline(always)]
    fn convergence_len(&self) -> usize {
        self.ctx.index.n_groups[0]
    }
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

    /// Accumulate coefficient contributions from one FE into the scratch buffer.
    ///
    /// For each observation i: scratch[i] += coef[start + fe[i]]
    #[inline]
    fn accumulate_fe_contributions(&mut self, fe_idx: usize, coef: &[f64]) {
        let start = self.ctx.index.coef_start[fe_idx];
        let fe = self.ctx.index.group_ids_for_fe(fe_idx);

        for (sum, &g) in self.scratch.iter_mut().zip(fe.iter()) {
            *sum += coef[start + g];
        }
    }

    /// Update coefficients for a single FE given the accumulated other-FE sums.
    ///
    /// For each group g in FE q:
    ///   coef_out[g] = (in_out[g] - Σ scratch[i] * w) / group_weight[g]
    #[inline]
    fn update_fe_coefficients(&self, fe_idx: usize, coef_out: &mut [f64]) {
        let start = self.ctx.index.coef_start[fe_idx];
        let n_groups = self.ctx.index.n_groups[fe_idx];
        let fe = self.ctx.index.group_ids_for_fe(fe_idx);
        let group_weights = self.ctx.group_weights_for_fe(fe_idx);

        // Initialize from in_out
        coef_out[start..start + n_groups]
            .copy_from_slice(&self.in_out[start..start + n_groups]);

        // Subtract accumulated other-FE contributions
        if self.ctx.weights.is_uniform {
            for (&g, &sum) in fe.iter().zip(self.scratch.iter()) {
                coef_out[start + g] -= sum;
            }
        } else {
            for ((&g, &sum), &w) in fe
                .iter()
                .zip(self.scratch.iter())
                .zip(self.ctx.weights.per_obs.iter())
            {
                coef_out[start + g] -= sum * w;
            }
        }

        // Normalize by group weights
        for (coef, &sw) in coef_out[start..start + n_groups]
            .iter_mut()
            .zip(group_weights.iter())
        {
            *coef /= sw;
        }
    }
}

impl Projector for MultiFEProjector<'_> {
    #[inline(always)]
    fn coef_len(&self) -> usize {
        self.ctx.index.n_coef
    }

    /// Project coefficients using reverse-order FE updates.
    ///
    /// For each FE q from (n_fe-1) down to 0:
    ///   1. Accumulate contributions from FEs before q (from coef_in)
    ///   2. Accumulate contributions from FEs after q (from coef_out, already computed)
    ///   3. Update coef_out for FE q
    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        let n_fe = self.ctx.index.n_fe;

        for q in (0..n_fe).rev() {
            // Reset scratch buffer
            self.scratch.fill(0.0);

            // Accumulate from FEs before q (use coef_in)
            for h in 0..q {
                self.accumulate_fe_contributions(h, coef_in);
            }

            // Accumulate from FEs after q (use coef_out, already computed)
            for h in (q + 1)..n_fe {
                self.accumulate_fe_contributions(h, coef_out);
            }

            // Update coefficients for FE q
            self.update_fe_coefficients(q, coef_out);
        }
    }

    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        let n_fe = self.ctx.index.n_fe;

        // Compute SSR: Σ (input[i] - Σ_q coef[fe_q[i]])²
        //
        // We iterate over FEs in the outer loop and observations in the inner loop.
        // This improves cache locality because:
        // 1. group_ids_for_fe(q) returns a contiguous slice for FE q
        // 2. We access the scratch buffer sequentially
        // 3. The coefficient array (typically small) stays in cache

        // Accumulate coefficient sums per observation using the scratch buffer
        self.scratch.fill(0.0);
        for q in 0..n_fe {
            let offset = self.ctx.index.coef_start[q];
            let fe_ids = self.ctx.index.group_ids_for_fe(q);
            for (sum, &g) in self.scratch.iter_mut().zip(fe_ids.iter()) {
                *sum += coef[offset + g];
            }
        }

        // Compute SSR from residuals
        self.input
            .iter()
            .zip(self.scratch.iter())
            .map(|(&x, &sum)| {
                let resid = x - sum;
                resid * resid
            })
            .sum()
    }

    #[inline(always)]
    fn convergence_len(&self) -> usize {
        self.ctx.index.n_coef - self.ctx.index.n_groups[self.ctx.index.n_fe - 1]
    }
}
