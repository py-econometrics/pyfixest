//! Core data types for accelerated demeaning.
//!
//! The main types are:
//! - [`FEStructure`]: Fixed effects indexing (which observation belongs to which group)
//! - [`Weights`]: Observation weights and their group-level aggregations
//! - [`FEInfo`]: Combines structure + weights for the demeaning algorithm
//! - [`FixestConfig`]: Algorithm parameters (tolerance, max iterations, etc.)

use ndarray::{ArrayView1, ArrayView2};

// =============================================================================
// Irons-Tuck Acceleration
// =============================================================================

/// Apply Irons-Tuck acceleration: given x, G(x), G(G(x)), compute accelerated update.
/// Returns `true` if converged (ssq == 0).
#[inline(always)]
pub fn irons_tuck_accelerate(x: &mut [f64], gx: &[f64], ggx: &[f64]) -> bool {
    let (vprod, ssq) = x
        .iter()
        .zip(gx.iter())
        .zip(ggx.iter())
        .map(|((&x_i, &gx_i), &ggx_i)| {
            let delta_gx = ggx_i - gx_i;
            let delta2_x = delta_gx - gx_i + x_i;
            (delta_gx * delta2_x, delta2_x * delta2_x)
        })
        .fold((0.0, 0.0), |(vp, sq), (dvp, dsq)| (vp + dvp, sq + dsq));

    if ssq == 0.0 {
        return true;
    }

    let coef = vprod / ssq;
    x.iter_mut()
        .zip(gx.iter())
        .zip(ggx.iter())
        .for_each(|((x_i, &gx_i), &ggx_i)| {
            *x_i = ggx_i - coef * (ggx_i - gx_i);
        });

    false
}

// =============================================================================
// Convergence Criteria
// =============================================================================

/// Returns true if a and b are converged (difference within tolerance).
#[inline]
pub fn converged(a: f64, b: f64, tol: f64) -> bool {
    let diff = (a - b).abs();
    (diff <= tol) || (diff / (0.1 + a.abs()) <= tol)
}

/// Returns `true` if NOT converged (should keep iterating).
#[inline]
pub fn should_continue(coef_old: &[f64], coef_new: &[f64], tol: f64) -> bool {
    coef_old.iter().zip(coef_new.iter()).any(|(&a, &b)| !converged(a, b, tol))
}

// =============================================================================
// FEStructure
// =============================================================================

/// Fixed effects indexing structure.
///
/// # Memory Layout
/// `group_ids` is a flat array where `group_ids[fe * n_obs + obs]` gives
/// the group ID for observation `obs` in fixed effect `fe`.
///
/// # Example
/// For 100 observations with 2 fixed effects (individual and time):
/// - `group_ids[0..100]` contains individual IDs for each observation
/// - `group_ids[100..200]` contains time period IDs for each observation
pub struct FEStructure {
    /// Number of observations
    pub n_obs: usize,
    /// Number of fixed effects
    pub n_fe: usize,
    /// Flat group IDs: index with `fe * n_obs + obs`
    pub group_ids: Vec<usize>,
    /// Number of groups per fixed effect
    pub groups_per_fe: Vec<usize>,
    /// Coefficient offset for each FE in the flat coefficient array
    pub coef_offset: Vec<usize>,
    /// Total coefficients across all FEs (sum of groups_per_fe)
    pub n_coef: usize,
}

impl FEStructure {
    /// Create FE structure from input array.
    ///
    /// # Arguments
    /// * `flist` - Fixed effect group IDs (shape: n_obs x n_fe)
    ///
    /// The number of groups per FE is computed automatically from max(group_id) + 1.
    pub fn new(flist: &ArrayView2<usize>) -> Self {
        let (n_obs, n_fe) = flist.dim();

        // Compute groups_per_fe: max group_id + 1 for each FE
        let groups_per_fe: Vec<usize> = (0..n_fe)
            .map(|j| flist.column(j).iter().max().unwrap_or(&0) + 1)
            .collect();

        // Compute coefficient offsets (cumulative sum of groups_per_fe)
        let mut coef_offset = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_offset[q] = coef_offset[q - 1] + groups_per_fe[q - 1];
        }
        let n_coef: usize = groups_per_fe.iter().sum();

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
            groups_per_fe,
            coef_offset,
            n_coef,
        }
    }

    /// Slice of group IDs for fixed effect `q`.
    #[inline(always)]
    pub fn group_ids_for_fe(&self, q: usize) -> &[usize] {
        let start = q * self.n_obs;
        &self.group_ids[start..start + self.n_obs]
    }
}

// =============================================================================
// Weights
// =============================================================================

/// Observation weights and their group-level aggregations.
///
/// For weighted least squares, observations have different weights.
/// `per_group` contains the sum of weights for each group, used for
/// computing weighted group means.
pub struct Weights {
    /// Per-observation weights (length: n_obs)
    pub per_obs: Vec<f64>,
    /// Sum of weights per group (length: n_coef)
    pub per_group: Vec<f64>,
    /// True if all observation weights are 1.0 (enables fast path)
    pub is_uniform: bool,
}

impl Weights {
    /// Create weights from observation weights and FE structure.
    pub fn new(weights: &ArrayView1<f64>, structure: &FEStructure) -> Self {
        let is_uniform = weights.iter().all(|&w| (w - 1.0).abs() < 1e-10);

        // Aggregate observation weights to group level
        let mut per_group = vec![0.0; structure.n_coef];
        for q in 0..structure.n_fe {
            let offset = structure.coef_offset[q];
            let fe_offset = q * structure.n_obs;
            for (i, &w) in weights.iter().enumerate() {
                let g = structure.group_ids[fe_offset + i];
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

    /// Slice of group weights for fixed effect `q`.
    #[inline(always)]
    pub fn group_weights_for_fe(&self, q: usize, structure: &FEStructure) -> &[f64] {
        let start = structure.coef_offset[q];
        let end = if q + 1 < structure.n_fe {
            structure.coef_offset[q + 1]
        } else {
            structure.n_coef
        };
        &self.per_group[start..end]
    }
}

// =============================================================================
// FEInfo
// =============================================================================

/// Complete fixed effects information for demeaning.
///
/// Combines [`FEStructure`] (indexing) with [`Weights`] (observation weights).
pub struct FEInfo {
    pub structure: FEStructure,
    pub weights: Weights,
}

impl FEInfo {
    /// Create FEInfo from input arrays.
    ///
    /// # Arguments
    /// * `flist` - Fixed effect group IDs (shape: n_obs x n_fe)
    /// * `weights` - Per-observation weights (length: n_obs)
    ///
    /// The number of groups per FE is computed automatically from max(group_id) + 1.
    pub fn new(flist: &ArrayView2<usize>, weights: &ArrayView1<f64>) -> Self {
        let structure = FEStructure::new(flist);
        let weights = Weights::new(weights, &structure);
        Self { structure, weights }
    }

    // -------------------------------------------------------------------------
    // Computation methods
    // -------------------------------------------------------------------------

    /// Compute weighted sums per group from input values.
    ///
    /// For each group g in each FE q:
    ///   result[coef_offset[q] + g] = sum over obs i in group g of: input[i] * weight[i]
    pub fn compute_in_out_from_input(&self, input: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.structure.n_coef];
        let n_obs = self.structure.n_obs;

        if self.weights.is_uniform {
            for q in 0..self.structure.n_fe {
                let offset = self.structure.coef_offset[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.structure.group_ids[fe_offset + i];
                    in_out[offset + g] += input[i];
                }
            }
        } else {
            for q in 0..self.structure.n_fe {
                let offset = self.structure.coef_offset[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.structure.group_ids[fe_offset + i];
                    in_out[offset + g] += input[i] * self.weights.per_obs[i];
                }
            }
        }

        in_out
    }

    /// Compute weighted sums per group from (input - subtract).
    pub fn compute_in_out(&self, input: &[f64], subtract: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.structure.n_coef];
        let n_obs = self.structure.n_obs;

        if self.weights.is_uniform {
            for q in 0..self.structure.n_fe {
                let offset = self.structure.coef_offset[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.structure.group_ids[fe_offset + i];
                    in_out[offset + g] += input[i] - subtract[i];
                }
            }
        } else {
            for q in 0..self.structure.n_fe {
                let offset = self.structure.coef_offset[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.structure.group_ids[fe_offset + i];
                    in_out[offset + g] += (input[i] - subtract[i]) * self.weights.per_obs[i];
                }
            }
        }

        in_out
    }

    /// Compute output = input - sum of FE coefficients for each observation.
    pub fn compute_output(&self, coef: &[f64], input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        let n_obs = self.structure.n_obs;
        for q in 0..self.structure.n_fe {
            let offset = self.structure.coef_offset[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = self.structure.group_ids[fe_offset + i];
                output[i] -= coef[offset + g];
            }
        }
    }

    /// Add FE coefficients to output for each observation.
    pub fn add_coef_to(&self, coef: &[f64], output: &mut [f64]) {
        let n_obs = self.structure.n_obs;
        for q in 0..self.structure.n_fe {
            let offset = self.structure.coef_offset[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = self.structure.group_ids[fe_offset + i];
                output[i] += coef[offset + g];
            }
        }
    }
}

// =============================================================================
// FixestConfig
// =============================================================================

/// Algorithm configuration parameters.
#[derive(Clone, Copy)]
pub struct FixestConfig {
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum iterations
    pub maxiter: usize,
    /// Warmup iterations before 2-FE sub-convergence (for 3+ FE)
    pub iter_warmup: usize,
    /// Iterations before projection after acceleration
    pub iter_proj_after_acc: usize,
    /// Iterations between grand acceleration steps
    pub iter_grand_acc: usize,
}


impl Default for FixestConfig {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            maxiter: 100_000,
            iter_warmup: 15,
            iter_proj_after_acc: 40,
            iter_grand_acc: 4,
        }
    }
}
