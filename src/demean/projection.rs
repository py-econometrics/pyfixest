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
//! Projectors are used with [`IronsTuckGrand`](crate::demean::accelerator::IronsTuckGrand)
//! which handles the iteration strategy.

use super::sweep::{GaussSeidelSweeper, TwoFESweeper};
use crate::demean::types::DemeanContext;
use smallvec::SmallVec;

// =============================================================================
// Projector Trait
// =============================================================================

/// A projection operation for fixed-effects demeaning.
///
/// Projectors hold all context needed for projection and provide the core
/// operations used by accelerators. All methods are called in tight loops
/// and should be optimized for performance.
pub trait Projector {
    /// Total number of coefficients this projector operates on.
    fn coef_len(&self) -> usize;

    /// Project coefficients: coef_in → coef_out.
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]);

    /// Compute the sum of squared residuals for the given coefficients.
    fn compute_ssr(&mut self, coef: &[f64]) -> f64;

    /// Range of coefficients to use for convergence checking.
    ///
    /// May be smaller than `0..coef_len()` when not all coefficients need checking.
    fn convergence_range(&self) -> std::ops::Range<usize>;
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
    // Dimensions
    n_obs: usize,
    n0: usize,
    n1: usize,

    // Sweepers for each direction
    /// Computes alpha from beta
    alpha_sweeper: TwoFESweeper<'a>,
    /// Computes beta from alpha
    beta_sweeper: TwoFESweeper<'a>,

    // Group ID pointers (needed for SSR computation)
    fe0_group_ids_ptr: *const usize,
    fe1_group_ids_ptr: *const usize,

    // Input data
    input: &'a [f64],

    // Scratch buffer for beta coefficients
    scratch: Vec<f64>,
}

impl<'a> TwoFEProjector<'a> {
    /// Create a new 2-FE projector.
    #[inline]
    pub fn new(ctx: &'a DemeanContext, coef_sums: &'a [f64], input: &'a [f64]) -> Self {
        let fe0_info = &ctx.fe_infos[0];
        let fe1_info = &ctx.fe_infos[1];
        let n0 = fe0_info.n_groups;
        let n1 = fe1_info.n_groups;
        let weights_ptr = ctx.weights.as_ref().map(|w| w.as_ptr());

        Self {
            n_obs: ctx.dims.n_obs,
            n0,
            n1,
            // alpha_sweeper: computes alpha from beta (out=fe0, other=fe1)
            alpha_sweeper: TwoFESweeper::new(
                ctx.dims.n_obs,
                weights_ptr,
                fe0_info,
                fe1_info,
                coef_sums,
                0, // alpha starts at offset 0
            ),
            // beta_sweeper: computes beta from alpha (out=fe1, other=fe0)
            beta_sweeper: TwoFESweeper::new(
                ctx.dims.n_obs,
                weights_ptr,
                fe1_info,
                fe0_info,
                coef_sums,
                n0, // beta starts at offset n0
            ),
            fe0_group_ids_ptr: fe0_info.group_ids.as_ptr(),
            fe1_group_ids_ptr: fe1_info.group_ids.as_ptr(),
            input,
            scratch: vec![0.0; n1],
        }
    }
}

impl Projector for TwoFEProjector<'_> {
    #[inline(always)]
    fn coef_len(&self) -> usize {
        self.n0 + self.n1
    }

    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        // Step 1: alpha_in -> beta (stored in scratch)
        self.beta_sweeper.sweep(&coef_in[..self.n0], &mut self.scratch);

        // Step 2: beta -> alpha_out
        self.alpha_sweeper.sweep(&self.scratch, &mut coef_out[..self.n0]);

        // Step 3: Copy beta to output
        coef_out[self.n0..self.n0 + self.n1].copy_from_slice(&self.scratch);
    }

    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        // Compute beta from alpha (updates self.scratch)
        self.beta_sweeper.sweep(&coef[..self.n0], &mut self.scratch);

        // Compute SSR: Σ (input[i] - alpha[fe0[i]] - beta[fe1[i]])²
        // Use 4x unrolling for better ILP
        let n_obs = self.n_obs;
        let chunks = n_obs / 4;
        let mut i = 0usize;
        let mut ssr = 0.0;

        // SAFETY: All pointer accesses are valid because:
        // - i < n_obs throughout (loop bounds ensure this)
        // - fe0_ptr, fe1_ptr point to arrays of length n_obs (from FixedEffectInfo)
        // - input_ptr points to array of length n_obs (from caller)
        // - group IDs (g0_*, g1_*) are always < n0 or < n1 respectively
        //   (invariant from DemeanContext construction)
        // - alpha_ptr points to coef with length >= n0, beta_ptr to scratch with length n1
        unsafe {
            let alpha_ptr = coef.as_ptr();
            let beta_ptr = self.scratch.as_ptr();
            let input_ptr = self.input.as_ptr();
            let fe0_ptr = self.fe0_group_ids_ptr;
            let fe1_ptr = self.fe1_group_ids_ptr;

            for _ in 0..chunks {
                let g0_0 = *fe0_ptr.add(i);
                let g0_1 = *fe0_ptr.add(i + 1);
                let g0_2 = *fe0_ptr.add(i + 2);
                let g0_3 = *fe0_ptr.add(i + 3);

                let g1_0 = *fe1_ptr.add(i);
                let g1_1 = *fe1_ptr.add(i + 1);
                let g1_2 = *fe1_ptr.add(i + 2);
                let g1_3 = *fe1_ptr.add(i + 3);

                debug_assert!(g0_0 < self.n0 && g0_1 < self.n0 && g0_2 < self.n0 && g0_3 < self.n0,
                    "FE0 group ID out of bounds: max({}, {}, {}, {}) >= n0 ({})",
                    g0_0, g0_1, g0_2, g0_3, self.n0);
                debug_assert!(g1_0 < self.n1 && g1_1 < self.n1 && g1_2 < self.n1 && g1_3 < self.n1,
                    "FE1 group ID out of bounds: max({}, {}, {}, {}) >= n1 ({})",
                    g1_0, g1_1, g1_2, g1_3, self.n1);

                let resid0 =
                    *input_ptr.add(i) - *alpha_ptr.add(g0_0) - *beta_ptr.add(g1_0);
                let resid1 =
                    *input_ptr.add(i + 1) - *alpha_ptr.add(g0_1) - *beta_ptr.add(g1_1);
                let resid2 =
                    *input_ptr.add(i + 2) - *alpha_ptr.add(g0_2) - *beta_ptr.add(g1_2);
                let resid3 =
                    *input_ptr.add(i + 3) - *alpha_ptr.add(g0_3) - *beta_ptr.add(g1_3);

                ssr += resid0 * resid0 + resid1 * resid1 + resid2 * resid2 + resid3 * resid3;
                i += 4;
            }

            // Handle remainder
            while i < n_obs {
                let g0 = *fe0_ptr.add(i);
                let g1 = *fe1_ptr.add(i);
                debug_assert!(g0 < self.n0, "FE0 group ID ({}) >= n0 ({})", g0, self.n0);
                debug_assert!(g1 < self.n1, "FE1 group ID ({}) >= n1 ({})", g1, self.n1);
                let resid = *input_ptr.add(i) - *alpha_ptr.add(g0) - *beta_ptr.add(g1);
                ssr += resid * resid;
                i += 1;
            }
        }
        ssr
    }

    #[inline(always)]
    fn convergence_range(&self) -> std::ops::Range<usize> {
        0..self.n0
    }
}

// =============================================================================
// MultiFEProjector
// =============================================================================

/// Projector for 3+ fixed effects.
///
/// Uses Gauss-Seidel block updates, processing FEs in reverse order
/// to match fixest's algorithm.
pub struct MultiFEProjector<'a> {
    ctx: &'a DemeanContext,
    input: &'a [f64],
    /// Pre-created sweepers for each FE (stored in reverse order for iteration).
    sweepers: Vec<GaussSeidelSweeper<'a>>,
    /// Precomputed (group_ids_ptr, coef_start) for each FE, used in SSR computation.
    /// SmallVec avoids heap allocation for typical 3-4 FE cases.
    fe_ptrs: SmallVec<[(*const usize, usize); 4]>,
}

impl<'a> MultiFEProjector<'a> {
    #[inline]
    pub fn new(ctx: &'a DemeanContext, coef_sums: &'a [f64], input: &'a [f64]) -> Self {
        // Pre-create sweepers in reverse order (how they're processed)
        let sweepers: Vec<_> = (0..ctx.dims.n_fe)
            .rev()
            .map(|q| GaussSeidelSweeper::new(ctx, coef_sums, q))
            .collect();

        // Precompute FE pointers for SSR computation (avoids per-call allocation)
        let fe_ptrs: SmallVec<[(*const usize, usize); 4]> = ctx
            .fe_infos
            .iter()
            .map(|fe| (fe.group_ids.as_ptr(), fe.coef_start))
            .collect();

        Self {
            ctx,
            input,
            sweepers,
            fe_ptrs,
        }
    }
}

impl Projector for MultiFEProjector<'_> {
    #[inline(always)]
    fn coef_len(&self) -> usize {
        self.ctx.dims.n_coef
    }

    #[inline(always)]
    fn project(&mut self, coef_in: &[f64], coef_out: &mut [f64]) {
        for sweeper in &self.sweepers {
            sweeper.sweep(coef_in, coef_out);
        }
    }

    #[inline(always)]
    fn compute_ssr(&mut self, coef: &[f64]) -> f64 {
        let n_obs = self.ctx.dims.n_obs;
        let coef_ptr = coef.as_ptr();
        let input_ptr = self.input.as_ptr();

        let mut ssr = 0.0;

        // SAFETY: All pointer accesses are valid because:
        // - i < n_obs throughout (loop bounds ensure this)
        // - group_ids_ptr for each FE points to array of length n_obs (from FixedEffectInfo)
        // - input_ptr points to array of length n_obs (from caller)
        // - group IDs are always < n_groups for their respective FE
        //   (invariant from DemeanContext construction)
        // - coef_start + g < coef.len() because coef_start is the FE's offset and
        //   g < n_groups for that FE (DemeanContext guarantees this layout)
        unsafe {
            // Main loop with 4x unrolling
            let chunks = n_obs / 4;
            let mut i = 0usize;

            for _ in 0..chunks {
                let mut sum0 = 0.0;
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                let mut sum3 = 0.0;

                for &(group_ids_ptr, coef_start) in &self.fe_ptrs {
                    let g0 = *group_ids_ptr.add(i);
                    let g1 = *group_ids_ptr.add(i + 1);
                    let g2 = *group_ids_ptr.add(i + 2);
                    let g3 = *group_ids_ptr.add(i + 3);

                    sum0 += *coef_ptr.add(coef_start + g0);
                    sum1 += *coef_ptr.add(coef_start + g1);
                    sum2 += *coef_ptr.add(coef_start + g2);
                    sum3 += *coef_ptr.add(coef_start + g3);
                }

                let resid0 = *input_ptr.add(i) - sum0;
                let resid1 = *input_ptr.add(i + 1) - sum1;
                let resid2 = *input_ptr.add(i + 2) - sum2;
                let resid3 = *input_ptr.add(i + 3) - sum3;

                ssr += resid0 * resid0 + resid1 * resid1 + resid2 * resid2 + resid3 * resid3;
                i += 4;
            }

            // Handle remainder
            while i < n_obs {
                let mut sum = 0.0;
                for &(group_ids_ptr, coef_start) in &self.fe_ptrs {
                    let g = *group_ids_ptr.add(i);
                    sum += *coef_ptr.add(coef_start + g);
                }
                let resid = *input_ptr.add(i) - sum;
                ssr += resid * resid;
                i += 1;
            }
        }

        ssr
    }

    #[inline(always)]
    fn convergence_range(&self) -> std::ops::Range<usize> {
        let n_fe = self.ctx.dims.n_fe;
        0..(self.ctx.dims.n_coef - self.ctx.fe_infos[n_fe - 1].n_groups)
    }
}
