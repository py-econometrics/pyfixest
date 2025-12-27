//! Core data types, traits, and default implementations for accelerated demeaning.

// =============================================================================
// Traits
// =============================================================================

/// Strategy for accelerating fixed-point iteration.
pub trait AccelerationStrategy: Send + Sync {
    /// Apply acceleration: given x, G(x), G(G(x)), compute accelerated update.
    /// Returns `true` if converged.
    fn accelerate(&self, x: &mut [f64], gx: &[f64], ggx: &[f64]) -> bool;
}

/// Criterion for checking convergence.
pub trait ConvergenceCriterion: Send + Sync {
    /// Returns `true` if NOT converged (should keep iterating).
    fn should_continue(&self, coef_old: &[f64], coef_new: &[f64], tol: f64) -> bool;

    /// Returns `true` if converged based on SSR change.
    fn ssr_converged(&self, ssr_old: f64, ssr_new: f64, tol: f64) -> bool {
        let _ = (ssr_old, ssr_new, tol);
        false
    }
}

// =============================================================================
// Default Implementations
// =============================================================================

/// Irons-Tuck acceleration (fixest default).
#[derive(Debug, Clone, Copy, Default)]
pub struct IronsTuck;

impl AccelerationStrategy for IronsTuck {
    #[inline(always)]
    fn accelerate(&self, x: &mut [f64], gx: &[f64], ggx: &[f64]) -> bool {
        let n = x.len();
        let mut vprod = 0.0;
        let mut ssq = 0.0;

        for i in 0..n {
            unsafe {
                let gx_i = *gx.get_unchecked(i);
                let ggx_i = *ggx.get_unchecked(i);
                let x_i = *x.get_unchecked(i);
                let delta_gx = ggx_i - gx_i;
                let delta2_x = delta_gx - gx_i + x_i;
                vprod += delta_gx * delta2_x;
                ssq += delta2_x * delta2_x;
            }
        }

        if ssq == 0.0 {
            return true;
        }

        let coef = vprod / ssq;
        for i in 0..n {
            unsafe {
                let gx_i = *gx.get_unchecked(i);
                let ggx_i = *ggx.get_unchecked(i);
                *x.get_unchecked_mut(i) = ggx_i - coef * (ggx_i - gx_i);
            }
        }

        false
    }
}

/// Fixest's convergence criterion (abs AND relative check).
#[derive(Debug, Clone, Copy, Default)]
pub struct FixestConvergence;

impl FixestConvergence {
    #[inline]
    fn continue_crit(a: f64, b: f64, diff_max: f64) -> bool {
        let diff = (a - b).abs();
        (diff > diff_max) && (diff / (0.1 + a.abs()) > diff_max)
    }

    #[inline]
    fn stopping_crit(a: f64, b: f64, diff_max: f64) -> bool {
        let diff = (a - b).abs();
        (diff < diff_max) || (diff / (0.1 + a.abs()) < diff_max)
    }
}

impl ConvergenceCriterion for FixestConvergence {
    fn should_continue(&self, coef_old: &[f64], coef_new: &[f64], tol: f64) -> bool {
        for i in 0..coef_old.len() {
            if Self::continue_crit(coef_old[i], coef_new[i], tol) {
                return true;
            }
        }
        false
    }

    fn ssr_converged(&self, ssr_old: f64, ssr_new: f64, tol: f64) -> bool {
        Self::stopping_crit(ssr_old, ssr_new, tol)
    }
}

// =============================================================================
// FEInfo
// =============================================================================

/// Pre-computed FE information with flat memory layout.
pub struct FEInfo {
    pub n_obs: usize,
    pub n_fe: usize,
    pub fe_ids: Vec<usize>,
    pub n_groups: Vec<usize>,
    pub coef_start: Vec<usize>,
    pub n_coef_total: usize,
    pub sum_weights: Vec<f64>,
    pub weights: Vec<f64>,
    pub is_unweighted: bool,
}

impl FEInfo {
    pub fn new(
        n_obs: usize,
        n_fe: usize,
        group_ids: &[usize],
        n_groups: &[usize],
        weights: &[f64],
    ) -> Self {
        let is_unweighted = weights.iter().all(|&w| (w - 1.0).abs() < 1e-10);

        let mut coef_start = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_start[q] = coef_start[q - 1] + n_groups[q - 1];
        }
        let n_coef_total: usize = n_groups.iter().sum();

        let mut fe_ids = vec![0usize; n_fe * n_obs];
        for i in 0..n_obs {
            for q in 0..n_fe {
                fe_ids[q * n_obs + i] = group_ids[i * n_fe + q];
            }
        }

        let mut sum_weights = vec![0.0; n_coef_total];
        for q in 0..n_fe {
            let start = coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = fe_ids[fe_offset + i];
                sum_weights[start + g] += weights[i];
            }
        }
        for s in &mut sum_weights {
            if *s == 0.0 {
                *s = 1.0;
            }
        }

        Self {
            n_obs,
            n_fe,
            fe_ids,
            n_groups: n_groups.to_vec(),
            coef_start,
            n_coef_total,
            sum_weights,
            weights: weights.to_vec(),
            is_unweighted,
        }
    }

    #[inline(always)]
    pub fn fe_ids_slice(&self, q: usize) -> &[usize] {
        let start = q * self.n_obs;
        &self.fe_ids[start..start + self.n_obs]
    }

    #[inline(always)]
    pub fn sum_weights_slice(&self, q: usize) -> &[f64] {
        let start = self.coef_start[q];
        let end = if q + 1 < self.n_fe {
            self.coef_start[q + 1]
        } else {
            self.n_coef_total
        };
        &self.sum_weights[start..end]
    }

    pub fn compute_in_out(&self, input: &[f64], output: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.n_coef_total];
        let n_obs = self.n_obs;

        if self.is_unweighted {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i] - output[i];
                }
            }
        } else {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += (input[i] - output[i]) * self.weights[i];
                }
            }
        }

        in_out
    }

    pub fn compute_output(&self, coef: &[f64], input: &[f64], output: &mut [f64]) {
        output.copy_from_slice(input);
        let n_obs = self.n_obs;
        for q in 0..self.n_fe {
            let start = self.coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = self.fe_ids[fe_offset + i];
                output[i] -= coef[start + g];
            }
        }
    }
}

// =============================================================================
// FixestConfig
// =============================================================================

#[derive(Clone, Copy)]
pub struct FixestConfig {
    pub tol: f64,
    pub maxiter: usize,
    pub iter_warmup: usize,
    pub iter_proj_after_acc: usize,
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
