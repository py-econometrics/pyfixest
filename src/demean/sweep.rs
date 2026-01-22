//! Block sweepers for fixed-effects demeaning.
//!
//! This module contains the low-level sweepers that encapsulate unsafe pointer
//! operations for the projection algorithms:
//!
//! - [`TwoFESweeper`]: For 2-FE case, computes one side's coefficients from the other
//! - [`GaussSeidelSweeper`]: For 3+ FE case, performs one block update in the Gauss-Seidel iteration

use crate::demean::types::{DemeanContext, FixedEffectInfo};
use smallvec::SmallVec;

// =============================================================================
// TwoFESweeper
// =============================================================================

/// Performs a single-direction sweep for 2-FE demeaning.
///
/// Each sweeper computes coefficients for one FE given the other FE's coefficients.
/// For a complete 2-FE iteration, use two instances:
/// - `alpha_sweeper`: computes alpha coefficients from beta
/// - `beta_sweeper`: computes beta coefficients from alpha
///
/// All data needed for the hot loop is precomputed at construction time
/// to minimize indirection during iteration.
pub(super) struct TwoFESweeper<'a> {
    n_obs: usize,
    n_groups: usize,

    // Per-observation weights (None = uniform)
    weights_ptr: Option<*const f64>,

    // This side's data
    out_groups_ptr: *const usize,
    inv_group_weights_ptr: *const f64,
    coef_sums_ptr: *const f64,

    // Other side's group IDs (for reading input coefficients)
    other_groups_ptr: *const usize,

    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> TwoFESweeper<'a> {
    /// Create a sweeper for computing `out_fe`'s coefficients from `other_fe`'s coefficients.
    #[inline]
    pub fn new(
        n_obs: usize,
        weights_ptr: Option<*const f64>,
        out_fe: &'a FixedEffectInfo,
        other_fe: &'a FixedEffectInfo,
        coef_sums: &'a [f64],
        out_coef_start: usize,
    ) -> Self {
        // Verify bounds before creating raw pointer
        debug_assert!(
            out_coef_start + out_fe.n_groups <= coef_sums.len(),
            "out_coef_start ({}) + n_groups ({}) exceeds coef_sums.len() ({})",
            out_coef_start,
            out_fe.n_groups,
            coef_sums.len()
        );

        // SAFETY: out_coef_start is the offset for this FE within coef_sums,
        // verified by debug_assert above and guaranteed by DemeanContext construction.
        let coef_sums_ptr = unsafe { coef_sums.as_ptr().add(out_coef_start) };

        Self {
            n_obs,
            n_groups: out_fe.n_groups,
            weights_ptr,
            out_groups_ptr: out_fe.group_ids.as_ptr(),
            inv_group_weights_ptr: out_fe.inv_group_weights.as_ptr(),
            coef_sums_ptr,
            other_groups_ptr: other_fe.group_ids.as_ptr(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute output coefficients from the other side's coefficients.
    ///
    /// Formula: `out[g] = (sums[g] - Σᵢ other[other_groups[i]] * w[i]) * inv_weights[g]`
    #[inline(always)]
    pub fn sweep(&self, other_coef: &[f64], out_coef: &mut [f64]) {
        debug_assert!(
            out_coef.len() >= self.n_groups,
            "out_coef.len() ({}) must be >= n_groups ({})",
            out_coef.len(),
            self.n_groups
        );

        let other_ptr = other_coef.as_ptr();
        let out_ptr = out_coef.as_mut_ptr();

        // SAFETY: All pointer operations are valid because:
        // - coef_sums_ptr points to n_groups elements (set in constructor)
        // - out_ptr has capacity n_groups (caller's responsibility, same as other_coef.len())
        // - inv_group_weights_ptr points to n_groups elements (from FixedEffectInfo)
        // - scatter_* methods only access indices < n_obs (loop bounds)
        // - group IDs are always < n_groups (invariant from DemeanContext construction)
        unsafe {
            // 1. Initialize from coef_sums
            std::ptr::copy_nonoverlapping(self.coef_sums_ptr, out_ptr, self.n_groups);

            // 2. Scatter-subtract
            match self.weights_ptr {
                None => self.scatter_uniform(other_ptr, out_ptr),
                Some(w_ptr) => self.scatter_weighted(other_ptr, out_ptr, w_ptr),
            }

            // 3. Normalize by inverse group weights (slice-based for auto-vectorization)
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, self.n_groups);
            let weights_slice =
                std::slice::from_raw_parts(self.inv_group_weights_ptr, self.n_groups);
            for (o, &w) in out_slice.iter_mut().zip(weights_slice.iter()) {
                *o *= w;
            }
        }
    }

    /// Scatter-subtract for uniform weights.
    #[inline(always)]
    unsafe fn scatter_uniform(&self, other_ptr: *const f64, out_ptr: *mut f64) {
        let out_groups = self.out_groups_ptr;
        let other_groups = self.other_groups_ptr;

        for i in 0..self.n_obs {
            let g_out = *out_groups.add(i);
            let g_other = *other_groups.add(i);
            debug_assert!(g_out < self.n_groups, "g_out ({}) >= n_groups ({})", g_out, self.n_groups);
            *out_ptr.add(g_out) -= *other_ptr.add(g_other);
        }
    }

    /// Scatter-subtract for weighted case.
    #[inline(always)]
    unsafe fn scatter_weighted(
        &self,
        other_ptr: *const f64,
        out_ptr: *mut f64,
        w_ptr: *const f64,
    ) {
        let out_groups = self.out_groups_ptr;
        let other_groups = self.other_groups_ptr;

        for i in 0..self.n_obs {
            let g_out = *out_groups.add(i);
            let g_other = *other_groups.add(i);
            debug_assert!(g_out < self.n_groups, "g_out ({}) >= n_groups ({})", g_out, self.n_groups);
            let w = *w_ptr.add(i);
            *out_ptr.add(g_out) -= *other_ptr.add(g_other) * w;
        }
    }
}

// =============================================================================
// OtherFEInfo
// =============================================================================

/// Precomputed info for accessing another FE's coefficients.
#[derive(Clone, Copy)]
pub(super) struct OtherFEInfo {
    /// Offset into coefficient array for this FE
    coef_start: usize,
    /// Pointer to group IDs for this FE
    group_ids_ptr: *const usize,
}

// =============================================================================
// GaussSeidelSweeper
// =============================================================================

/// Performs Gauss-Seidel block sweeps for multi-FE demeaning.
///
/// All data needed for the hot loop is precomputed at construction time
/// to minimize indirection during iteration.
pub(super) struct GaussSeidelSweeper<'a> {
    // This FE's cached data
    n_obs: usize,
    coef_start: usize,
    n_groups: usize,
    group_ids_ptr: *const usize,
    inv_group_weights_ptr: *const f64,
    coef_sums_ptr: *const f64,

    // Weight info: None = uniform (unweighted), Some = weighted
    weights_ptr: Option<*const f64>,

    // Other FEs' info (precomputed to avoid fe_infos lookup in hot loop)
    // SmallVec avoids heap allocation for typical 2-5 FE cases (max 4 other FEs)
    /// FEs processed before this one (read from coef_in)
    other_before: SmallVec<[OtherFEInfo; 4]>,
    /// FEs processed after this one (read from coef_out)
    other_after: SmallVec<[OtherFEInfo; 4]>,

    /// Marker to tie the struct's lifetime to the borrowed data.
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> GaussSeidelSweeper<'a> {
    #[inline]
    pub fn new(ctx: &'a DemeanContext, coef_sums: &'a [f64], q: usize) -> Self {
        let fe = &ctx.fe_infos[q];

        // Precompute other FEs' info
        let other_before: SmallVec<[OtherFEInfo; 4]> = (0..q)
            .map(|h| {
                let fe_h = &ctx.fe_infos[h];
                OtherFEInfo {
                    coef_start: fe_h.coef_start,
                    group_ids_ptr: fe_h.group_ids.as_ptr(),
                }
            })
            .collect();

        let other_after: SmallVec<[OtherFEInfo; 4]> = ((q + 1)..ctx.dims.n_fe)
            .map(|h| {
                let fe_h = &ctx.fe_infos[h];
                OtherFEInfo {
                    coef_start: fe_h.coef_start,
                    group_ids_ptr: fe_h.group_ids.as_ptr(),
                }
            })
            .collect();

        // Verify bounds before creating raw pointer
        debug_assert!(
            fe.coef_start + fe.n_groups <= coef_sums.len(),
            "coef_start ({}) + n_groups ({}) exceeds coef_sums.len() ({})",
            fe.coef_start,
            fe.n_groups,
            coef_sums.len()
        );

        // SAFETY: fe.coef_start is the offset for this FE within coef_sums,
        // verified by debug_assert above and guaranteed by DemeanContext construction.
        let coef_sums_ptr = unsafe { coef_sums.as_ptr().add(fe.coef_start) };

        Self {
            n_obs: ctx.dims.n_obs,
            coef_start: fe.coef_start,
            n_groups: fe.n_groups,
            group_ids_ptr: fe.group_ids.as_ptr(),
            inv_group_weights_ptr: fe.inv_group_weights.as_ptr(),
            coef_sums_ptr,
            weights_ptr: ctx.weights.as_ref().map(|w| w.as_ptr()),
            other_before,
            other_after,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform one Gauss-Seidel block update for this FE.
    #[inline(always)]
    pub fn sweep(&self, coef_in: &[f64], coef_out: &mut [f64]) {
        debug_assert!(
            coef_out.len() >= self.coef_start + self.n_groups,
            "coef_out.len() ({}) must be >= coef_start + n_groups ({})",
            coef_out.len(),
            self.coef_start + self.n_groups
        );

        let coef_in_ptr = coef_in.as_ptr();
        let coef_out_ptr = coef_out.as_mut_ptr();

        // SAFETY: All pointer operations are valid because:
        // - coef_start + n_groups <= coef_out.len() (caller provides full coefficient array)
        // - coef_sums_ptr points to n_groups elements (set in constructor)
        // - inv_group_weights_ptr points to n_groups elements (from FixedEffectInfo)
        // - scatter_* methods only access indices < n_obs (loop bounds)
        // - group IDs are always < n_groups (invariant from DemeanContext construction)
        // - other_before/other_after coef_starts are valid offsets into coef arrays
        unsafe {
            // 1. Initialize from coef_sums
            let out_start = coef_out_ptr.add(self.coef_start);
            std::ptr::copy_nonoverlapping(self.coef_sums_ptr, out_start, self.n_groups);

            // 2. Scatter-subtract
            match self.weights_ptr {
                None => self.scatter_uniform(coef_in_ptr, coef_out_ptr, out_start),
                Some(w_ptr) => self.scatter_weighted(coef_in_ptr, coef_out_ptr, out_start, w_ptr),
            }

            // 3. Normalize by inverse group weights (slice-based for auto-vectorization)
            let out_slice = std::slice::from_raw_parts_mut(out_start, self.n_groups);
            let weights_slice =
                std::slice::from_raw_parts(self.inv_group_weights_ptr, self.n_groups);
            for (o, &w) in out_slice.iter_mut().zip(weights_slice.iter()) {
                *o *= w;
            }
        }
    }

    /// Scatter-subtract for uniform weights.
    #[inline(always)]
    unsafe fn scatter_uniform(
        &self,
        coef_in_ptr: *const f64,
        coef_out_ptr: *mut f64,
        out_start: *mut f64,
    ) {
        let group_ids = self.group_ids_ptr;

        for i in 0..self.n_obs {
            let sum = self.accumulate_other_effects(i, coef_in_ptr, coef_out_ptr);
            let g = *group_ids.add(i);
            debug_assert!(g < self.n_groups, "g ({}) >= n_groups ({})", g, self.n_groups);
            *out_start.add(g) -= sum;
        }
    }

    /// Scatter-subtract for weighted case.
    #[inline(always)]
    unsafe fn scatter_weighted(
        &self,
        coef_in_ptr: *const f64,
        coef_out_ptr: *mut f64,
        out_start: *mut f64,
        w_ptr: *const f64,
    ) {
        let group_ids = self.group_ids_ptr;

        for i in 0..self.n_obs {
            let sum = self.accumulate_other_effects(i, coef_in_ptr, coef_out_ptr);
            let g = *group_ids.add(i);
            debug_assert!(g < self.n_groups, "g ({}) >= n_groups ({})", g, self.n_groups);
            let w = *w_ptr.add(i);
            *out_start.add(g) -= sum * w;
        }
    }

    /// Accumulate coefficient contributions from all other FEs.
    ///
    /// This is the innermost hot loop - kept minimal for best inlining.
    #[inline(always)]
    unsafe fn accumulate_other_effects(
        &self,
        i: usize,
        coef_in_ptr: *const f64,
        coef_out_ptr: *mut f64,
    ) -> f64 {
        let mut sum = 0.0;

        // FEs before this one: read from coef_in
        for other in &self.other_before {
            let g = *other.group_ids_ptr.add(i);
            sum += *coef_in_ptr.add(other.coef_start + g);
        }

        // FEs after this one: read from coef_out (already updated)
        for other in &self.other_after {
            let g = *other.group_ids_ptr.add(i);
            sum += *coef_out_ptr.add(other.coef_start + g);
        }

        sum
    }
}
