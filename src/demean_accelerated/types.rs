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
//! - [`FixedEffectsIndex`]: Maps observations to their group IDs for each FE
//! - [`ObservationWeights`]: Per-observation and per-group weight sums
//! - [`DemeanContext`]: Combines index + weights, provides scatter/gather operations
//! - [`FixestConfig`]: Algorithm parameters (tolerance, max iterations, etc.)

use ndarray::{ArrayView1, ArrayView2};
use std::ops::Range;

// =============================================================================
// FixedEffectsIndex
// =============================================================================

/// Index mapping observations to fixed effect groups.
///
/// # Purpose
///
/// Maps each observation to its group ID for each fixed effect. For example,
/// observation 42 might belong to individual 7 and time period 3.
///
/// # Memory Layout
///
/// Group IDs are stored in column-major order for cache efficiency during iteration:
/// ```text
/// group_ids = [fe0_obs0, fe0_obs1, ..., fe0_obsN, fe1_obs0, fe1_obs1, ..., fe1_obsN, ...]
///              |-------- FE 0 ----------|         |-------- FE 1 ----------|
/// ```
///
/// Access pattern: `group_ids[fe_index * n_obs + obs_index]`
///
/// # Example
///
/// ```text
/// 1000 observations, 2 fixed effects (individual, year):
/// - n_groups = [100, 10]      // 100 individuals, 10 years
/// - coef_start = [0, 100]     // individuals at 0..100, years at 100..110
/// - n_coef = 110              // total coefficients
/// ```
pub struct FixedEffectsIndex {
    /// Number of observations (N).
    pub n_obs: usize,

    /// Number of fixed effects (e.g., 2 for individual + time).
    pub n_fe: usize,

    /// Flat group IDs in column-major order.
    /// Index with `fe * n_obs + obs` to get the group ID for observation `obs` in FE `fe`.
    pub group_ids: Vec<usize>,

    /// Number of groups in each fixed effect.
    /// Example: `[100, 10]` means FE 0 has 100 groups, FE 1 has 10 groups.
    pub n_groups: Vec<usize>,

    /// Starting index in coefficient arrays for each FE.
    /// Example: `[0, 100]` means FE 0 coefficients are at indices 0..100,
    /// FE 1 coefficients are at indices 100..110.
    pub coef_start: Vec<usize>,

    /// Total number of coefficients (sum of `n_groups`).
    pub n_coef: usize,
}

impl FixedEffectsIndex {
    /// Create a fixed effects index from the input array.
    ///
    /// # Arguments
    ///
    /// * `flist` - Fixed effect group IDs with shape `(n_obs, n_fe)`.
    ///   Each row is one observation, each column is one fixed effect.
    ///   Values must be 0-indexed group IDs.
    ///
    /// # Computed Fields
    ///
    /// - `n_groups`: Computed as `max(group_id) + 1` for each FE
    /// - `coef_start`: Cumulative sum of `n_groups`
    /// - `group_ids`: Transposed to column-major order for cache efficiency
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `n_obs == 0` or `n_fe == 0`.
    pub fn new(flist: &ArrayView2<usize>) -> Self {
        let (n_obs, n_fe) = flist.dim();

        debug_assert!(n_obs > 0, "Cannot create FixedEffectsIndex with 0 observations");
        debug_assert!(n_fe > 0, "Cannot create FixedEffectsIndex with 0 fixed effects");

        // Compute n_groups: max group_id + 1 for each FE
        let n_groups: Vec<usize> = (0..n_fe)
            .map(|j| flist.column(j).iter().max().unwrap_or(&0) + 1)
            .collect();

        // Compute coefficient start indices (cumulative sum of n_groups)
        let mut coef_start = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_start[q] = coef_start[q - 1] + n_groups[q - 1];
        }
        let n_coef: usize = n_groups.iter().sum();

        // Transpose group_ids from row-major (obs, fe) to column-major (fe, obs)
        // This layout is better for the inner loops which iterate over observations
        let mut group_ids = vec![0usize; n_fe * n_obs];
        for q in 0..n_fe {
            for (i, &g) in flist.column(q).iter().enumerate() {
                group_ids[q * n_obs + i] = g;
            }
        }

        Self {
            n_obs,
            n_fe,
            group_ids,
            n_groups,
            coef_start,
            n_coef,
        }
    }

    /// Get the group IDs for all observations in fixed effect `fe`.
    ///
    /// Returns a slice of length `n_obs` where `result[i]` is the group ID
    /// for observation `i` in this fixed effect.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let individual_ids = index.group_ids_for_fe(0);  // [7, 3, 7, 12, ...]
    /// let year_ids = index.group_ids_for_fe(1);        // [0, 1, 0, 2, ...]
    /// ```
    #[inline(always)]
    pub fn group_ids_for_fe(&self, fe: usize) -> &[usize] {
        let start = fe * self.n_obs;
        &self.group_ids[start..start + self.n_obs]
    }

    /// Get the coefficient index range for fixed effect `fe`.
    ///
    /// Returns the range of indices in coefficient arrays that correspond
    /// to this fixed effect's groups.
    #[inline(always)]
    pub fn coef_range_for_fe(&self, fe: usize) -> Range<usize> {
        let start = self.coef_start[fe];
        let end = if fe + 1 < self.n_fe {
            self.coef_start[fe + 1]
        } else {
            self.n_coef
        };
        start..end
    }
}

// =============================================================================
// ObservationWeights
// =============================================================================

/// Observation weights and their aggregation to group level.
///
/// # Purpose
///
/// In weighted least squares, observations have different weights (e.g., inverse
/// variance weights). To compute weighted group means, we need:
///
/// 1. Per-observation weights for the numerator: `Σ(weight[i] * value[i])`
/// 2. Per-group weight sums for the denominator: `Σ(weight[i])` for each group
///
/// # Uniform Weights Fast Path
///
/// When all weights are 1.0 (unweighted regression), `is_uniform = true` enables
/// optimized code paths that skip multiplication by weights.
pub struct ObservationWeights {
    /// Weight for each observation (length: `n_obs`).
    /// Used when scattering values to coefficient space.
    pub per_obs: Vec<f64>,

    /// Sum of observation weights for each group (length: `n_coef`).
    /// Used as denominator when computing group means.
    /// Layout matches coefficient space: `[fe0_group0, ..., fe0_groupK, fe1_group0, ...]`.
    pub per_group: Vec<f64>,

    /// True if all observation weights are 1.0 (enables fast path).
    pub is_uniform: bool,
}

impl ObservationWeights {
    /// Create observation weights from the input array.
    ///
    /// # Arguments
    ///
    /// * `weights` - Per-observation weights (length: `n_obs`)
    /// * `index` - Fixed effects index (needed to aggregate weights to groups)
    ///
    /// # Computed Fields
    ///
    /// - `is_uniform`: True if all weights are 1.0 (within floating-point tolerance)
    /// - `per_group`: Sum of observation weights for each group
    pub fn new(weights: &ArrayView1<f64>, index: &FixedEffectsIndex) -> Self {
        // Tolerance for detecting uniform weights (all 1.0).
        // Using 1e-10 to account for floating-point representation errors
        // while being strict enough to catch intentionally non-uniform weights.
        const UNIFORM_WEIGHT_TOL: f64 = 1e-10;
        let is_uniform = weights.iter().all(|&w| (w - 1.0).abs() < UNIFORM_WEIGHT_TOL);

        // Aggregate observation weights to group level
        let mut per_group = vec![0.0; index.n_coef];
        for q in 0..index.n_fe {
            let offset = index.coef_start[q];
            let fe_offset = q * index.n_obs;
            for (i, &w) in weights.iter().enumerate() {
                let g = index.group_ids[fe_offset + i];
                per_group[offset + g] += w;
            }
        }

        // Avoid division by zero for empty groups
        for w in &mut per_group {
            if *w == 0.0 {
                *w = 1.0;
            }
        }

        Self {
            per_obs: weights.to_vec(),
            per_group,
            is_uniform,
        }
    }
}

// =============================================================================
// DemeanContext
// =============================================================================

/// Complete context for fixed effects demeaning operations.
///
/// # Purpose
///
/// Combines the fixed effects index (which observation belongs to which groups)
/// with observation weights. Provides the core scatter/gather operations needed
/// by the iterative demeaning algorithm.
///
/// # Operations
///
/// The demeaning algorithm repeatedly:
///
/// 1. **Scatter**: Aggregate residuals from observations to group coefficients
/// 2. **Gather**: Subtract group coefficients from observations
///
/// These operations transform data between observation space (N values) and
/// coefficient space (`n_coef` values).
///
/// # Example Usage
///
/// ```ignore
/// let ctx = DemeanContext::new(&flist, &weights);
///
/// // Scatter input to coefficient space
/// let coef_sums = ctx.scatter_to_coefficients(&input);
///
/// // Compute group means: coef[g] = coef_sums[g] / group_weight[g]
/// // ... (done in solver)
/// ```
pub struct DemeanContext {
    /// Fixed effects index (observation → group mapping).
    pub index: FixedEffectsIndex,

    /// Observation weights and group-level aggregations.
    pub weights: ObservationWeights,
}

impl DemeanContext {
    /// Create a demeaning context from input arrays.
    ///
    /// # Arguments
    ///
    /// * `flist` - Fixed effect group IDs with shape `(n_obs, n_fe)`
    /// * `weights` - Per-observation weights (length: `n_obs`)
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `weights.len() != flist.nrows()`.
    pub fn new(flist: &ArrayView2<usize>, weights: &ArrayView1<f64>) -> Self {
        debug_assert_eq!(
            weights.len(),
            flist.nrows(),
            "weights length ({}) must match number of observations ({})",
            weights.len(),
            flist.nrows()
        );

        let index = FixedEffectsIndex::new(flist);
        let weights = ObservationWeights::new(weights, &index);
        Self { index, weights }
    }

    /// Get the weight sums for all groups in fixed effect `fe`.
    #[inline(always)]
    pub fn group_weights_for_fe(&self, fe: usize) -> &[f64] {
        &self.weights.per_group[self.index.coef_range_for_fe(fe)]
    }

    // =========================================================================
    // Scatter/Gather Operations
    // =========================================================================

    /// Scatter values from observation space to coefficient space.
    ///
    /// Computes weighted sums of `values` for each group in each FE.
    /// Returns a vector of length `n_coef` with the aggregated sums.
    #[inline]
    pub fn scatter_to_coefficients(&self, values: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.index.n_coef];
        self.scatter_inner(values, None, &mut result);
        result
    }

    /// Scatter residuals from observation space to coefficient space.
    ///
    /// Like [`scatter_to_coefficients`], but first subtracts `baseline` from `values`.
    /// Computes: `Σ (values[i] - baseline[i]) * weight[i]` for each group.
    #[inline]
    pub fn scatter_residuals(&self, values: &[f64], baseline: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.index.n_coef];
        self.scatter_inner(values, Some(baseline), &mut result);
        result
    }

    /// Gather coefficients to observation space and add to output.
    ///
    /// For each observation, looks up its coefficient for each FE and adds to output.
    /// Computes: `output[i] += Σ_q coef[offset_q + fe_q[i]]`
    #[inline]
    pub fn gather_and_add(&self, coef: &[f64], output: &mut [f64]) {
        for q in 0..self.index.n_fe {
            let offset = self.index.coef_start[q];
            let fe_ids = self.index.group_ids_for_fe(q);
            for (i, &g) in fe_ids.iter().enumerate() {
                output[i] += coef[offset + g];
            }
        }
    }

    /// Inner scatter implementation with optional baseline subtraction.
    ///
    /// Handles both uniform and non-uniform weights with optimized code paths.
    #[inline(always)]
    fn scatter_inner(&self, values: &[f64], baseline: Option<&[f64]>, result: &mut [f64]) {
        for q in 0..self.index.n_fe {
            let offset = self.index.coef_start[q];
            let fe_ids = self.index.group_ids_for_fe(q);

            match (self.weights.is_uniform, baseline) {
                (true, None) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i];
                    }
                }
                (true, Some(base)) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i] - base[i];
                    }
                }
                (false, None) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += values[i] * self.weights.per_obs[i];
                    }
                }
                (false, Some(base)) => {
                    for (i, &g) in fe_ids.iter().enumerate() {
                        result[offset + g] += (values[i] - base[i]) * self.weights.per_obs[i];
                    }
                }
            }
        }
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
pub struct FixestConfig {
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
}

impl Default for FixestConfig {
    /// Default values match R's fixest package for consistency.
    fn default() -> Self {
        Self {
            // Default tolerance matches fixest's `fixest_options("demean_tol")`
            tol: 1e-6,
            // Generous iteration limit to handle difficult convergence cases
            maxiter: 100_000,
            // Warmup iterations before 2-FE sub-convergence (fixest default)
            iter_warmup: 15,
            // Post-acceleration projection starts after this many iterations
            iter_proj_after_acc: 40,
            // Grand acceleration frequency (every N iterations)
            iter_grand_acc: 4,
            // SSR convergence check frequency
            ssr_check_interval: 40,
        }
    }
}

// =============================================================================
// ConvergenceState
// =============================================================================

/// Whether the iterative algorithm has converged.
///
/// Used throughout the demeaning module to represent convergence state
/// in a self-documenting way, avoiding ambiguous boolean returns.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceState {
    /// Algorithm has converged; iteration can stop.
    Converged,
    /// Algorithm has not yet converged; continue iterating.
    NotConverged,
}
