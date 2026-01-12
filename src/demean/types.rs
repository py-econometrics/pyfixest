//! Core data types for accelerated fixed effects demeaning.
//!
//! # Overview
//!
//! Fixed effects demeaning removes group means from data. For example, with
//! individual and time fixed effects, we remove both individual-specific and
//! time-specific means from each observation.
//!
//! # Two Spaces
//!
//! The algorithm works in two "spaces":
//!
//! - **Observation space**: Length N (number of observations)
//!   - Input data, output data, residuals
//!
//! - **Coefficient space**: Length = sum of groups across all FEs
//!   - One coefficient per group per FE
//!   - Example: 1000 individuals + 10 years = 1010 coefficients
//!   - Stored flat: `[individual_0, ..., individual_999, year_0, ..., year_9]`
//!
//! # Core Operations
//!
//! 1. **Scatter** (obs → coef): Aggregate weighted values from observations to group sums
//! 2. **Gather** (coef → obs): Look up each observation's group coefficients and combine
//!
//! These operations are the building blocks of the iterative demeaning algorithm.
//!
//! # Main Types
//!
//! - [`Dimensions`]: Problem shape (n_obs, n_fe, n_coef)
//! - [`Weights`]: Observation-level weights (None = uniform weights)
//! - [`FixedEffectInfo`]: Per-FE group IDs and weights
//! - [`DemeanContext`]: Combines all of the above, provides scatter/gather operations
//! - [`FixestConfig`]: Algorithm parameters (tolerance, max iterations, etc.)

use ndarray::{Array2, ArrayView1, ArrayView2};

// =============================================================================
// Dimensions
// =============================================================================

/// Problem dimensions for fixed effects demeaning.
///
/// The algorithm operates in two spaces:
/// - **Observation space**: length `n_obs` (input/output data)
/// - **Coefficient space**: length `n_coef` (one coefficient per group per FE)
///
/// # Example
///
/// With 10,000 observations, 500 firms, and 20 years:
/// - `n_obs = 10_000`
/// - `n_fe = 2`
/// - `n_coef = 520` (500 firm coefficients + 20 year coefficients)
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dimensions {
    /// Number of observations (N).
    pub n_obs: usize,
    /// Number of fixed effects (Q). E.g., 2 for firm + year.
    pub n_fe: usize,
    /// Total coefficients: sum of group counts across all FEs.
    pub n_coef: usize,
}


// =============================================================================
// FixedEffectInfo
// =============================================================================

/// Information for a single fixed effect.
///
/// Each fixed effect (e.g., firm, year) has its own group structure.
/// This struct holds the mapping from observations to groups and the
/// precomputed weight sums needed for computing group means.
///
/// # Coefficient Layout
///
/// Coefficients for all FEs are stored in a single flat array:
/// ```text
/// [FE0_group0, ..., FE0_groupK, FE1_group0, ..., FE1_groupM, ...]
/// ```
/// The `coef_start` field gives the offset where this FE's coefficients begin.
#[derive(Clone, Debug)]
pub(crate) struct FixedEffectInfo {
    /// Number of groups in this FE. E.g., 500 firms.
    pub n_groups: usize,
    /// Starting index in coefficient arrays for this FE.
    pub coef_start: usize,
    /// Group ID for each observation (length: `n_obs`).
    /// `group_ids[i]` gives the group index (0..n_groups) for observation i.
    pub group_ids: Vec<usize>,
    /// Inverse of group weights (length: `n_groups`).
    /// Precomputed as `1.0 / sum_of_observation_weights_per_group` to replace
    /// division with multiplication in hot loops. For unweighted case, this is
    /// `1.0 / count_of_observations_per_group`.
    pub inv_group_weights: Vec<f64>,
}

// =============================================================================
// DemeanContext
// =============================================================================

/// Complete context for fixed effects demeaning operations.
///
/// Combines problem dimensions, observation weights, and per-FE information.
/// Provides the core scatter/gather operations used by the iterative algorithm.
///
/// # Construction
///
/// Use [`DemeanContext::new`] to create a context from input arrays. The context
/// is reused across multiple columns being demeaned.
///
/// # FE Ordering
///
/// Fixed effects can optionally be reordered by size (largest first) via the
/// `reorder_fe` parameter. When enabled, this matches fixest's behavior and
/// can improve convergence for some datasets.
///
/// # Uniform Weights Fast Path
///
/// When `weights` is `None`, all observations are equally weighted. This enables
/// optimized code paths that skip weight multiplication in hot loops.
///
/// # Operations
///
/// - [`apply_design_matrix_t`](Self::apply_design_matrix_t): Scatter values to coefficient space
/// - [`apply_design_matrix`](Self::apply_design_matrix): Gather coefficients to observation space
pub struct DemeanContext {
    /// Problem dimensions.
    pub(crate) dims: Dimensions,
    /// Observation-level weights (length: `n_obs`). None means uniform weights (unweighted case).
    pub(crate) weights: Option<Vec<f64>>,
    /// Per-fixed-effect information (in internal/reordered order).
    pub(crate) fe_infos: Vec<FixedEffectInfo>,
    /// Mapping from internal FE index to original FE index.
    /// `fe_order[q]` gives the original column index for internal FE `q`.
    /// Used to reorder coefficients back to original order when returning.
    pub(crate) fe_order: Vec<usize>,
}

impl DemeanContext {
    /// Create a demeaning context from input arrays.
    ///
    /// # Arguments
    ///
    /// * `flist` - Fixed effect group IDs with shape `(n_obs, n_fe)`.
    ///   Each row is one observation, each column is one fixed effect.
    ///   Values must be 0-indexed group IDs.
    /// * `weights` - Per-observation weights (length: `n_obs`), or None for unweighted.
    /// * `reorder_fe` - If true, reorder FEs by size (largest first) before demeaning.
    ///   This can improve convergence for some datasets.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `flist` has zero rows or columns
    /// - `weights.len() != flist.nrows()`
    ///
    /// # Empty Groups
    ///
    /// Groups with no observations (e.g., sparse group IDs) are handled by setting
    /// their weight to 1, matching fixest's approach. Since no observation belongs
    /// to these groups, their coefficients are never used in computations.
    pub fn new(
        flist: &ArrayView2<usize>,
        weights: Option<&ArrayView1<f64>>,
        reorder_fe: bool,
    ) -> Self {
        let (n_obs, n_fe) = flist.dim();

        assert!(n_obs > 0, "Cannot create DemeanContext with 0 observations");
        assert!(n_fe > 0, "Cannot create DemeanContext with 0 fixed effects");
        if let Some(w) = weights {
            assert_eq!(
                w.len(),
                n_obs,
                "weights length ({}) must match number of observations ({})",
                w.len(),
                n_obs
            );
        }

        // Compute n_groups for each FE (max group_id + 1)
        // Panics if any column is empty (which shouldn't happen with n_obs > 0)
        let n_groups_original: Vec<usize> = (0..n_fe)
            .map(|j| {
                flist
                    .column(j)
                    .iter()
                    .max()
                    .expect("FE column should not be empty when n_obs > 0")
                    + 1
            })
            .collect();

        // Optionally reorder FEs by size (largest first)
        let order: Vec<usize> = if reorder_fe && n_fe > 1 {
            let mut indices: Vec<usize> = (0..n_fe).collect();
            indices.sort_by_key(|&i| std::cmp::Reverse(n_groups_original[i]));
            indices
        } else {
            (0..n_fe).collect()
        };

        // Compute dimensions
        let n_groups: Vec<usize> = order.iter().map(|&i| n_groups_original[i]).collect();
        let mut coef_starts = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_starts[q] = coef_starts[q - 1] + n_groups[q - 1];
        }
        let n_coef: usize = n_groups.iter().sum();

        let dims = Dimensions { n_obs, n_fe, n_coef };

        // Build observation weights (None if uniform)
        let obs_weights = weights.map(|w| w.to_vec());

        // Build per-FE info
        let mut fe_infos = Vec::with_capacity(n_fe);
        for q in 0..n_fe {
            let original_col = order[q];

            // Extract group IDs for this FE
            let group_ids: Vec<usize> = flist.column(original_col).iter().copied().collect();

            // Aggregate observation weights to group level
            let mut group_weights = vec![0.0; n_groups[q]];
            match &obs_weights {
                Some(w) => {
                    for (i, &g) in group_ids.iter().enumerate() {
                        group_weights[g] += w[i];
                    }
                }
                None => {
                    // Unweighted: count observations per group
                    for &g in group_ids.iter() {
                        group_weights[g] += 1.0;
                    }
                }
            }

            // Handle empty groups (weight=0) by setting weight to 1, matching fixest's approach.
            // This is defensive programming - empty groups are never accessed since no
            // observation belongs to them, but this prevents any potential division by zero.
            for w in &mut group_weights {
                if *w == 0.0 {
                    *w = 1.0;
                }
            }

            let inv_group_weights: Vec<f64> = group_weights.iter().map(|&w| 1.0 / w).collect();

            fe_infos.push(FixedEffectInfo {
                n_groups: n_groups[q],
                coef_start: coef_starts[q],
                group_ids,
                inv_group_weights,
            });
        }

        Self {
            dims,
            weights: obs_weights,
            fe_infos,
            fe_order: order,
        }
    }

    // =========================================================================
    // Design Matrix Operations (D and Dᵀ)
    // =========================================================================

    /// Apply transpose of design matrix: Dᵀ · values.
    ///
    /// Computes weighted sums of `values` for each group in each FE,
    /// writing the result to `out`. The buffer is zeroed before accumulation.
    ///
    /// # Arguments
    ///
    /// * `values` - Input values in observation space (length: `n_obs`)
    /// * `out` - Output buffer in coefficient space (length: `n_coef`)
    ///
    /// # Example
    ///
    /// With 4 observations, 2 firms (FE0), 2 years (FE1):
    ///
    /// ```text
    /// values = [10, 20, 30, 40]
    /// firm   = [ 0,  0,  1,  1]
    /// year   = [ 0,  1,  0,  1]
    ///
    /// out = [10+20, 30+40, 10+30, 20+40] = [30, 70, 40, 60]
    ///       |-- FE0 --|  |-- FE1 --|
    /// ```
    #[inline]
    pub fn apply_design_matrix_t(&self, values: &[f64], out: &mut [f64]) {
        debug_assert_eq!(
            out.len(),
            self.dims.n_coef,
            "output buffer length ({}) must match n_coef ({})",
            out.len(),
            self.dims.n_coef
        );
        out.fill(0.0);

        for fe in &self.fe_infos {
            let offset = fe.coef_start;
            match &self.weights {
                None => {
                    for (i, &g) in fe.group_ids.iter().enumerate() {
                        out[offset + g] += values[i];
                    }
                }
                Some(w) => {
                    for (i, &g) in fe.group_ids.iter().enumerate() {
                        out[offset + g] += values[i] * w[i];
                    }
                }
            }
        }
    }

    /// Apply design matrix and add to output: output += D · coef.
    ///
    /// For each observation, looks up its coefficient for each FE and adds to output.
    ///
    /// # Arguments
    ///
    /// * `coef` - Coefficients in coefficient space (length: `n_coef`)
    /// * `output` - Output buffer in observation space (length: `n_obs`), accumulated into
    #[inline]
    pub fn apply_design_matrix(&self, coef: &[f64], output: &mut [f64]) {
        for fe in &self.fe_infos {
            let offset = fe.coef_start;
            for (i, &g) in fe.group_ids.iter().enumerate() {
                output[i] += coef[offset + g];
            }
        }
    }

    /// Reorder coefficients from internal order to original FE order.
    ///
    /// The input `coef` is in internal order (potentially reordered by size).
    /// Returns coefficients in the original FE column order from the input flist.
    #[must_use]
    pub fn reorder_coef_to_original(&self, coef: &[f64]) -> Vec<f64> {
        let n_fe = self.dims.n_fe;

        // Build inverse mapping: original_fe_index -> internal_fe_index
        let mut internal_idx = vec![0usize; n_fe];
        for (q, &orig) in self.fe_order.iter().enumerate() {
            internal_idx[orig] = q;
        }

        // Reorder coefficients
        let mut out = Vec::with_capacity(self.dims.n_coef);
        for orig_fe in 0..n_fe {
            let q = internal_idx[orig_fe];
            let fe = &self.fe_infos[q];
            let start = fe.coef_start;
            let end = start + fe.n_groups;
            out.extend_from_slice(&coef[start..end]);
        }
        out
    }
}

// =============================================================================
// FixestConfig
// =============================================================================

/// Algorithm configuration parameters.
///
/// These parameters control the convergence behavior of the iterative
/// demeaning algorithm. The defaults match R's fixest package.
#[derive(Clone, Copy)]
pub(crate) struct FixestConfig {
    /// Convergence tolerance for coefficient changes.
    pub tol: f64,

    /// Maximum number of iterations before giving up.
    pub maxiter: usize,

    /// Warmup iterations before 2-FE sub-convergence (for 3+ FE).
    /// During warmup, all FEs are updated together.
    pub iter_warmup: usize,

    /// Iterations before applying projection after acceleration.
    pub iter_proj_after_acc: usize,

    /// Iterations between grand acceleration steps.
    pub iter_grand_acc: usize,

    /// Iterations between SSR-based convergence checks.
    pub ssr_check_interval: usize,

    /// Whether to reorder fixed effects by size (largest first) before demeaning.
    /// When true, FEs are processed in order of decreasing group count, which
    /// can improve convergence for some datasets. Default is false.
    pub reorder_fe: bool,
}

impl Default for FixestConfig {
    /// Default values match R's fixest package for consistency.
    fn default() -> Self {
        Self {
            tol: 1e-6,
            maxiter: 100_000,
            iter_warmup: 15,
            iter_proj_after_acc: 40,
            iter_grand_acc: 4,
            ssr_check_interval: 40,
            reorder_fe: false,
        }
    }
}

// =============================================================================
// ConvergenceState
// =============================================================================

/// Whether the iterative algorithm has converged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ConvergenceState {
    /// Algorithm has converged; iteration can stop.
    Converged,
    /// Algorithm has not yet converged; continue iterating.
    #[default]
    NotConverged,
}

// =============================================================================
// DemeanResult
// =============================================================================

/// Result of a demeaning operation (single column).
#[derive(Debug, Clone)]
pub(crate) struct DemeanResult {
    /// Demeaned data (length: `n_obs`).
    pub demeaned: Vec<f64>,

    /// Fixed effect coefficients in original FE order (length: `n_coef`).
    /// Laid out as `[FE0_coefs..., FE1_coefs..., ...]` where FE0, FE1, etc.
    /// are in the original input order (not reordered).
    pub fe_coefficients: Vec<f64>,

    /// Convergence state.
    pub convergence: ConvergenceState,

    /// Number of iterations used (0 for closed-form solutions).
    #[allow(dead_code)]
    pub iterations: usize,
}

// =============================================================================
// DemeanMultiResult
// =============================================================================

/// Result of demeaning multiple columns.
///
/// Returned by the [`demean`](super::demean) function which processes
/// multiple columns in parallel.
pub(crate) struct DemeanMultiResult {
    /// Demeaned data with shape `(n_samples, n_features)`.
    pub demeaned: Array2<f64>,

    /// Fixed effect coefficients with shape `(n_coef, n_features)`.
    /// Each column contains the FE coefficients for the corresponding input column.
    pub fe_coefficients: Array2<f64>,

    /// True if all columns converged successfully.
    pub success: bool,
}
