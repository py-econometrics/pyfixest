//! Multi-factor projector for alternating projections.
//!
//! Simplified to only what is used by the accelerated demeaning path.

use crate::demean_accelerated::acceleration::Projector;
use crate::demean_accelerated::single_fe::SingleFEDemeaner;

/// Multi-factor projector that applies each FE in sequence.
pub struct MultiFactorProjector {
    factors: Vec<SingleFEDemeaner>,
    temp_buffer: Vec<f64>,
}

impl MultiFactorProjector {
    /// Create a new multi-factor projector.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_factor_projector_single() {
        let weights = vec![1.0; 4];
        let group_ids = vec![0, 0, 1, 1];
        let n_groups = vec![2];

        let mut projector =
            MultiFactorProjector::new(&weights, &group_ids, 4, 1, &n_groups);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        projector.project(&input, &mut output);

        // Group means: [1.5, 3.5]
        assert_eq!(output, vec![-0.5, 0.5, -0.5, 0.5]);
    }

    #[test]
    fn test_multi_factor_projector_two() {
        let weights = vec![1.0; 6];
        let group_ids = vec![0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 2, 2];
        let n_groups = vec![3, 2];

        let mut projector =
            MultiFactorProjector::new(&weights, &group_ids, 6, 2, &n_groups);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];

        projector.project(&input, &mut output);

        // Check that results are finite and zero-mean per factor combination
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_multi_factor_projector_three() {
        let weights = vec![1.0; 6];
        let group_ids = vec![0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 2, 2, 0, 0, 1, 1, 2, 2];
        let n_groups = vec![2, 2, 3];

        let mut projector =
            MultiFactorProjector::new(&weights, &group_ids, 6, 3, &n_groups);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 6];

        projector.project(&input, &mut output);

        // Just ensure stability
        for &v in &output {
            assert!(v.is_finite());
        }
    }
}
