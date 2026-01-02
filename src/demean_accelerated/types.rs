//! Core data types and default implementations for accelerated demeaning.

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
// Fixest Convergence Criterion
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

    /// Compute in_out coefficients from input - subtract (subtrahend defaults to 0).
    pub fn compute_in_out(&self, input: &[f64], subtract: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.n_coef_total];
        let n_obs = self.n_obs;

        if self.is_unweighted {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i] - subtract[i];
                }
            }
        } else {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += (input[i] - subtract[i]) * self.weights[i];
                }
            }
        }

        in_out
    }

    /// Compute in_out coefficients directly from input (no subtraction).
    pub fn compute_in_out_from_input(&self, input: &[f64]) -> Vec<f64> {
        let mut in_out = vec![0.0; self.n_coef_total];
        let n_obs = self.n_obs;

        if self.is_unweighted {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i];
                }
            }
        } else {
            for q in 0..self.n_fe {
                let start = self.coef_start[q];
                let fe_offset = q * n_obs;
                for i in 0..n_obs {
                    let g = self.fe_ids[fe_offset + i];
                    in_out[start + g] += input[i] * self.weights[i];
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

    /// Add coefficient values to output (expand coefficients to observations).
    pub fn add_coef_to(&self, coef: &[f64], output: &mut [f64]) {
        let n_obs = self.n_obs;
        for q in 0..self.n_fe {
            let start = self.coef_start[q];
            let fe_offset = q * n_obs;
            for i in 0..n_obs {
                let g = self.fe_ids[fe_offset + i];
                output[i] += coef[start + g];
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
