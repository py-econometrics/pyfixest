//! General multi-FE demeaner with adaptive iteration budgeting.
//!
//! For 3+ fixed effects, uses a combination of:
//! - Irons-Tuck acceleration on fine timescale
//! - Grand acceleration on coarse timescale
//! - SSR-based convergence checking
//! - Adaptive fallback strategy

use crate::demean_accelerated::acceleration::{
    GrandAcceleration, IronsTuckAcceleration, Projector, StepResult,
};
use crate::demean_accelerated::single_fe::SingleFEDemeaner;

/// Multi-factor projector that applies each FE in sequence.
pub struct MultiFactorProjector {
    factors: Vec<SingleFEDemeaner>,
    temp_buffer: Vec<f64>,
}

impl MultiFactorProjector {
    /// Create a new multi-factor projector.
    ///
    /// # Arguments
    /// * `sample_weights` - Weight for each observation
    /// * `group_ids` - Flattened array of group IDs (n_samples * n_factors, row-major)
    /// * `n_samples` - Number of observations
    /// * `n_factors` - Number of fixed effects
    /// * `n_groups_per_factor` - Number of groups for each factor
    pub fn new(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups_per_factor: &[usize],
    ) -> Self {
        let mut factors = Vec::with_capacity(n_factors);

        for j in 0..n_factors {
            // Extract group IDs for this factor
            let factor_group_ids: Vec<usize> = (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .collect();

            factors.push(SingleFEDemeaner::new(
                sample_weights,
                &factor_group_ids,
                n_groups_per_factor[j],
            ));
        }

        Self {
            factors,
            temp_buffer: vec![0.0; n_samples],
        }
    }

    /// Get number of factors.
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }
}

impl Projector for MultiFactorProjector {
    fn project(&mut self, input: &[f64], output: &mut [f64]) {
        if self.factors.is_empty() {
            output.copy_from_slice(input);
            return;
        }

        // First factor: input -> output
        self.factors[0].demean(input, output);

        // Subsequent factors: alternate between output and temp_buffer
        let mut use_temp_as_output = true;
        for factor in self.factors.iter_mut().skip(1) {
            if use_temp_as_output {
                factor.demean(output, &mut self.temp_buffer);
            } else {
                factor.demean(&self.temp_buffer, output);
            }
            use_temp_as_output = !use_temp_as_output;
        }

        // If final result is in temp_buffer, copy to output
        if !use_temp_as_output {
            output.copy_from_slice(&self.temp_buffer);
        }
    }
}

/// Configuration for GeneralDemeaner.
#[derive(Clone)]
pub struct GeneralDemeanerConfig {
    /// Tolerance for convergence
    pub tol: f64,
    /// Maximum iterations
    pub maxiter: usize,
    /// Warmup iterations before checking convergence
    pub warmup_iters: usize,
    /// Interval for grand acceleration snapshots
    pub grand_accel_interval: usize,
    /// Interval for SSR-based convergence check
    pub ssr_check_interval: usize,
}

impl Default for GeneralDemeanerConfig {
    fn default() -> Self {
        Self {
            tol: 1e-8,
            maxiter: 100_000,
            warmup_iters: 15,
            grand_accel_interval: 15,
            ssr_check_interval: 40,
        }
    }
}

/// General demeaner for 3+ fixed effects.
///
/// Uses IronsTuckAcceleration with GrandAcceleration and
/// adaptive iteration budgeting for robust convergence.
pub struct GeneralDemeaner {
    acceleration: IronsTuckAcceleration<MultiFactorProjector>,
    grand_accel: GrandAcceleration,
    config: GeneralDemeanerConfig,
    n_samples: usize,
    /// Previous SSR for SSR-based convergence
    prev_ssr: f64,
}

impl GeneralDemeaner {
    /// Create a new GeneralDemeaner.
    pub fn new(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups_per_factor: &[usize],
        config: GeneralDemeanerConfig,
    ) -> Self {
        let projector = MultiFactorProjector::new(
            sample_weights,
            group_ids,
            n_samples,
            n_factors,
            n_groups_per_factor,
        );

        let acceleration = IronsTuckAcceleration::new(projector, n_samples);
        let grand_accel = GrandAcceleration::new(n_samples, config.grand_accel_interval);

        Self {
            acceleration,
            grand_accel,
            config,
            n_samples,
            prev_ssr: f64::MAX,
        }
    }

    /// Demean the input vector.
    ///
    /// Returns (demeaned_output, converged).
    pub fn demean(&mut self, input: &[f64], output: &mut [f64]) -> bool {
        self.acceleration.set_initial(input);
        self.prev_ssr = f64::MAX;
        self.grand_accel.reset();

        let mut converged = false;

        for iter in 0..self.config.maxiter {
            // Determine if this is an acceleration step (every 3rd iteration after first)
            let should_accelerate = iter % 3 == 0 && iter > 0;

            let step_result = self.acceleration.step(should_accelerate);

            if step_result == StepResult::NumericallyConverged {
                converged = true;
                break;
            }

            // Check element-wise convergence after warmup
            if iter >= self.config.warmup_iters && self.acceleration.is_converged(self.config.tol)
            {
                converged = true;
                break;
            }

            // Grand acceleration: record snapshots and apply when ready
            if self.grand_accel.should_record(iter) {
                self.grand_accel.record(self.acceleration.get_result());

                if self.grand_accel.can_apply() {
                    // Get mutable access to current iterate and apply grand acceleration
                    let current = self.acceleration.get_result().to_vec();
                    let mut accelerated = current;
                    self.grand_accel.apply(&mut accelerated);
                    self.acceleration.set_initial(&accelerated);
                }
            }

            // SSR-based convergence check every N iterations
            if iter > 0 && iter % self.config.ssr_check_interval == 0 {
                let ssr = self.compute_ssr(input);
                let ssr_change = (self.prev_ssr - ssr).abs() / (self.prev_ssr.abs() + 1e-10);

                if ssr_change < self.config.tol {
                    converged = true;
                    break;
                }

                self.prev_ssr = ssr;
            }
        }

        output.copy_from_slice(self.acceleration.get_result());
        converged
    }

    /// Compute sum of squared residuals.
    fn compute_ssr(&self, input: &[f64]) -> f64 {
        let current = self.acceleration.get_result();
        let mut ssr = 0.0;
        for i in 0..self.n_samples {
            let residual = input[i] - current[i];
            ssr += residual * residual;
        }
        ssr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_factor_projector_single() {
        let weights = vec![1.0; 6];
        let group_ids = vec![0, 0, 0, 1, 1, 1]; // Single factor
        let n_groups = vec![2];

        let mut projector = MultiFactorProjector::new(&weights, &group_ids, 6, 1, &n_groups);

        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = vec![0.0; 6];

        projector.project(&input, &mut output);

        // Should be same as SingleFEDemeaner
        // Group 0 mean = 2.0, Group 1 mean = 20.0
        assert!((output[0] - (-1.0)).abs() < 1e-10);
        assert!((output[1] - 0.0).abs() < 1e-10);
        assert!((output[2] - 1.0).abs() < 1e-10);
        assert!((output[3] - (-10.0)).abs() < 1e-10);
        assert!((output[4] - 0.0).abs() < 1e-10);
        assert!((output[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_factor_projector_two() {
        let weights = vec![1.0; 4];
        // Two factors: FE1 = [0,0,1,1], FE2 = [0,1,0,1]
        let group_ids = vec![0, 0, 0, 1, 1, 0, 1, 1]; // Flattened row-major
        let n_groups = vec![2, 2];

        let mut projector = MultiFactorProjector::new(&weights, &group_ids, 4, 2, &n_groups);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        projector.project(&input, &mut output);

        // Just verify it produces finite output
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_multi_factor_projector_three() {
        let weights = vec![1.0; 8];
        // Three factors
        let group_ids = vec![
            0, 0, 0, // obs 0: FE1=0, FE2=0, FE3=0
            0, 0, 1, // obs 1: FE1=0, FE2=0, FE3=1
            0, 1, 0, // obs 2: FE1=0, FE2=1, FE3=0
            0, 1, 1, // obs 3: FE1=0, FE2=1, FE3=1
            1, 0, 0, // obs 4: FE1=1, FE2=0, FE3=0
            1, 0, 1, // obs 5: FE1=1, FE2=0, FE3=1
            1, 1, 0, // obs 6: FE1=1, FE2=1, FE3=0
            1, 1, 1, // obs 7: FE1=1, FE2=1, FE3=1
        ];
        let n_groups = vec![2, 2, 2];

        let mut projector = MultiFactorProjector::new(&weights, &group_ids, 8, 3, &n_groups);

        let input: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let mut output = vec![0.0; 8];

        projector.project(&input, &mut output);

        // Just verify it produces finite output
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_general_demeaner_converges() {
        let weights = vec![1.0; 8];
        // Three factors (same as above)
        let group_ids = vec![
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
        ];
        let n_groups = vec![2, 2, 2];

        let config = GeneralDemeanerConfig {
            tol: 1e-8,
            maxiter: 1000,
            warmup_iters: 5,
            grand_accel_interval: 10,
            ssr_check_interval: 20,
        };

        let mut demeaner =
            GeneralDemeaner::new(&weights, &group_ids, 8, 3, &n_groups, config);

        let input: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let mut output = vec![0.0; 8];

        let converged = demeaner.demean(&input, &mut output);

        assert!(converged, "GeneralDemeaner should converge");

        // Verify output is finite
        for &v in &output {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_general_demeaner_respects_maxiter() {
        let weights = vec![1.0; 4];
        let group_ids = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let n_groups = vec![2, 2];

        let config = GeneralDemeanerConfig {
            tol: 1e-20, // Very tight tolerance - unlikely to converge
            maxiter: 10,
            warmup_iters: 5,
            grand_accel_interval: 5,
            ssr_check_interval: 5,
        };

        let mut demeaner = GeneralDemeaner::new(&weights, &group_ids, 4, 2, &n_groups, config);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        // Should return (even if not converged) after maxiter
        let _converged = demeaner.demean(&input, &mut output);

        // Output should still be finite
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
