//! Two fixed effects demeaner with coefficient-based iteration.
//!
//! For the common 2-FE case, we can optimize memory usage by storing
//! the second FE coefficients in a smaller buffer (n_groups[1] instead of n_obs).

use crate::demean_accelerated::acceleration::Projector;
use crate::demean_accelerated::single_fe::SingleFEDemeaner;

/// Two fixed effects demeaner with coefficient-based second FE.
///
/// This optimizes the common 2-FE case by:
/// 1. Using SingleFEDemeaner for the first FE (full observation vector)
/// 2. Storing second FE coefficients in a smaller n_groups[1]-length buffer
///
/// This reduces memory bandwidth when n_groups[1] << n_obs.
pub struct TwoFEDemeaner {
    /// First FE demeaner (operates on full observation vector)
    fe1: SingleFEDemeaner,
    /// Group IDs for second FE
    fe2_group_ids: Vec<usize>,
    /// Sample weights
    sample_weights: Vec<f64>,
    /// Pre-computed sum of weights per group for FE2
    fe2_group_weight_sums: Vec<f64>,
    /// Working buffer for FE2 weighted sums per group
    fe2_group_weighted_sums: Vec<f64>,
    /// Working buffer for intermediate results (n_obs length)
    temp_buffer: Vec<f64>,
}

impl TwoFEDemeaner {
    /// Create a new TwoFEDemeaner.
    ///
    /// # Arguments
    /// * `sample_weights` - Weight for each observation
    /// * `fe1_group_ids` - Group IDs for first FE (0-indexed)
    /// * `fe2_group_ids` - Group IDs for second FE (0-indexed)
    /// * `n_groups_fe1` - Number of groups for first FE
    /// * `n_groups_fe2` - Number of groups for second FE
    pub fn new(
        sample_weights: &[f64],
        fe1_group_ids: &[usize],
        fe2_group_ids: &[usize],
        n_groups_fe1: usize,
        n_groups_fe2: usize,
    ) -> Self {
        let n_samples = sample_weights.len();
        debug_assert_eq!(fe1_group_ids.len(), n_samples);
        debug_assert_eq!(fe2_group_ids.len(), n_samples);

        // Create first FE demeaner
        let fe1 = SingleFEDemeaner::new(sample_weights, fe1_group_ids, n_groups_fe1);

        // Pre-compute sum of weights per group for FE2
        let mut fe2_group_weight_sums = vec![0.0; n_groups_fe2];
        for i in 0..n_samples {
            let gid = fe2_group_ids[i];
            fe2_group_weight_sums[gid] += sample_weights[i];
        }

        Self {
            fe1,
            fe2_group_ids: fe2_group_ids.to_vec(),
            sample_weights: sample_weights.to_vec(),
            fe2_group_weight_sums,
            fe2_group_weighted_sums: vec![0.0; n_groups_fe2],
            temp_buffer: vec![0.0; n_samples],
        }
    }

    /// Project onto the intersection of both FE subspaces.
    ///
    /// Performs one sweep: demean by FE1, then demean by FE2.
    fn project_impl(&mut self, input: &[f64], output: &mut [f64]) {
        let n_samples = input.len();

        // Step 1: Demean by FE1 (input -> temp_buffer)
        self.fe1.demean(input, &mut self.temp_buffer);

        // Step 2: Demean by FE2 (temp_buffer -> output)
        // Using coefficient-based approach for FE2

        // Reset weighted sums
        self.fe2_group_weighted_sums.fill(0.0);

        // Accumulate weighted sums per group
        for i in 0..n_samples {
            let gid = self.fe2_group_ids[i];
            self.fe2_group_weighted_sums[gid] += self.sample_weights[i] * self.temp_buffer[i];
        }

        // Compute demeaned values
        for i in 0..n_samples {
            let gid = self.fe2_group_ids[i];
            let group_mean =
                self.fe2_group_weighted_sums[gid] / self.fe2_group_weight_sums[gid];
            output[i] = self.temp_buffer[i] - group_mean;
        }
    }
}

impl Projector for TwoFEDemeaner {
    fn project(&mut self, input: &[f64], output: &mut [f64]) {
        self.project_impl(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_fe_basic() {
        // 6 observations, 2 groups for each FE
        let weights = vec![1.0; 6];
        // FE1: groups [0,0,0,1,1,1]
        let fe1_ids = vec![0, 0, 0, 1, 1, 1];
        // FE2: groups [0,1,0,1,0,1] (alternating)
        let fe2_ids = vec![0, 1, 0, 1, 0, 1];

        let mut demeaner = TwoFEDemeaner::new(&weights, &fe1_ids, &fe2_ids, 2, 2);

        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut output = vec![0.0; 6];

        demeaner.project(&input, &mut output);

        // After demeaning, the sum within each FE group should be approximately zero
        // (may not be exact after just one sweep for 2 FEs)

        // Verify output has reasonable values (not NaN or extreme)
        for &v in &output {
            assert!(v.is_finite(), "Output contains non-finite value: {}", v);
            assert!(v.abs() < 100.0, "Output value too large: {}", v);
        }
    }

    #[test]
    fn test_two_fe_converges() {
        // Test that repeated projection converges
        let weights = vec![1.0; 8];
        let fe1_ids = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let fe2_ids = vec![0, 1, 0, 1, 0, 1, 0, 1];

        let mut demeaner = TwoFEDemeaner::new(&weights, &fe1_ids, &fe2_ids, 4, 2);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut current = input.clone();
        let mut next = vec![0.0; 8];

        // Iterate until convergence
        for _ in 0..100 {
            demeaner.project(&current, &mut next);
            std::mem::swap(&mut current, &mut next);
        }

        // After convergence, another projection should give same result
        demeaner.project(&current, &mut next);

        let max_diff = current
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        assert!(
            max_diff < 1e-10,
            "Did not converge after 100 iterations, max_diff = {}",
            max_diff
        );
    }

    #[test]
    fn test_two_fe_weighted() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let fe1_ids = vec![0, 0, 1, 1];
        let fe2_ids = vec![0, 1, 0, 1];

        let mut demeaner = TwoFEDemeaner::new(&weights, &fe1_ids, &fe2_ids, 2, 2);

        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 4];

        demeaner.project(&input, &mut output);

        // Verify output is finite
        for &v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_two_fe_implements_projector() {
        // Verify TwoFEDemeaner can be used as a Projector
        fn use_projector<P: Projector>(p: &mut P, input: &[f64], output: &mut [f64]) {
            p.project(input, output);
        }

        let weights = vec![1.0; 4];
        let fe1_ids = vec![0, 0, 1, 1];
        let fe2_ids = vec![0, 1, 0, 1];

        let mut demeaner = TwoFEDemeaner::new(&weights, &fe1_ids, &fe2_ids, 2, 2);

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        use_projector(&mut demeaner, &input, &mut output);

        // Just verify it ran without panic
        assert!(output.iter().all(|&v| v.is_finite()));
    }
}
