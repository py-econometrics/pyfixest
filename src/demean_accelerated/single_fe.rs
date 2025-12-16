//! Single fixed effect demeaner with O(n) closed-form solution.
//!
//! When there's only one fixed effect, demeaning is a direct computation
//! without iteration: subtract the weighted group mean from each observation.

/// Single fixed effect demeaner using closed-form solution.
///
/// For a single FE, demeaning is simply:
/// `output[i] = input[i] - weighted_mean(group[i])`
///
/// where `weighted_mean(g) = sum(w[i] * x[i] for i in group g) / sum(w[i] for i in group g)`
///
/// This is O(n) with no iteration required.
pub struct SingleFEDemeaner {
    /// Sample weights for each observation
    sample_weights: Vec<f64>,
    /// Group ID for each observation
    group_ids: Vec<usize>,
    /// Pre-computed sum of weights per group
    group_weight_sums: Vec<f64>,
    /// Working buffer for weighted sums per group
    group_weighted_sums: Vec<f64>,
    /// Number of groups
    n_groups: usize,
}

impl SingleFEDemeaner {
    /// Create a new SingleFEDemeaner.
    ///
    /// # Arguments
    /// * `sample_weights` - Weight for each observation
    /// * `group_ids` - Group ID for each observation (0-indexed, contiguous)
    /// * `n_groups` - Total number of groups
    ///
    /// # Panics
    /// Panics if `sample_weights.len() != group_ids.len()`
    pub fn new(sample_weights: &[f64], group_ids: &[usize], n_groups: usize) -> Self {
        debug_assert_eq!(sample_weights.len(), group_ids.len());

        let n_samples = sample_weights.len();

        // Pre-compute sum of weights per group
        let mut group_weight_sums = vec![0.0; n_groups];
        for i in 0..n_samples {
            let gid = group_ids[i];
            group_weight_sums[gid] += sample_weights[i];
        }

        Self {
            sample_weights: sample_weights.to_vec(),
            group_ids: group_ids.to_vec(),
            group_weight_sums,
            group_weighted_sums: vec![0.0; n_groups],
            n_groups,
        }
    }

    /// Demean the input vector in O(n) time.
    ///
    /// # Arguments
    /// * `input` - Input values to demean
    /// * `output` - Output buffer for demeaned values
    ///
    /// # Panics
    /// Panics if input/output lengths don't match the number of samples.
    pub fn demean(&mut self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), self.sample_weights.len());
        debug_assert_eq!(output.len(), input.len());

        let n_samples = input.len();

        // Reset weighted sums
        self.group_weighted_sums.fill(0.0);

        // Pass 1: Accumulate weighted sums per group
        for i in 0..n_samples {
            let gid = self.group_ids[i];
            self.group_weighted_sums[gid] += self.sample_weights[i] * input[i];
        }

        // Pass 2: Compute demeaned values
        for i in 0..n_samples {
            let gid = self.group_ids[i];
            let group_mean = self.group_weighted_sums[gid] / self.group_weight_sums[gid];
            output[i] = input[i] - group_mean;
        }
    }

    /// Demean in-place, modifying the input buffer.
    pub fn demean_inplace(&mut self, data: &mut [f64]) {
        debug_assert_eq!(data.len(), self.sample_weights.len());

        let n_samples = data.len();

        // Reset weighted sums
        self.group_weighted_sums.fill(0.0);

        // Pass 1: Accumulate weighted sums per group
        for i in 0..n_samples {
            let gid = self.group_ids[i];
            self.group_weighted_sums[gid] += self.sample_weights[i] * data[i];
        }

        // Pass 2: Subtract group means in-place
        for i in 0..n_samples {
            let gid = self.group_ids[i];
            let group_mean = self.group_weighted_sums[gid] / self.group_weight_sums[gid];
            data[i] -= group_mean;
        }
    }

    /// Get the number of groups.
    #[inline]
    pub fn n_groups(&self) -> usize {
        self.n_groups
    }

    /// Get the number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.sample_weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_fe_basic() {
        // 6 observations, 2 groups
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let group_ids = vec![0, 0, 0, 1, 1, 1];
        let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];

        let mut demeaner = SingleFEDemeaner::new(&weights, &group_ids, 2);
        let mut output = vec![0.0; 6];
        demeaner.demean(&input, &mut output);

        // Group 0 mean = (1+2+3)/3 = 2.0
        // Group 1 mean = (10+20+30)/3 = 20.0
        assert!((output[0] - (-1.0)).abs() < 1e-10); // 1 - 2 = -1
        assert!((output[1] - 0.0).abs() < 1e-10);    // 2 - 2 = 0
        assert!((output[2] - 1.0).abs() < 1e-10);   // 3 - 2 = 1
        assert!((output[3] - (-10.0)).abs() < 1e-10); // 10 - 20 = -10
        assert!((output[4] - 0.0).abs() < 1e-10);    // 20 - 20 = 0
        assert!((output[5] - 10.0).abs() < 1e-10);  // 30 - 20 = 10
    }

    #[test]
    fn test_single_fe_weighted() {
        // 4 observations, 2 groups, with weights
        let weights = vec![1.0, 2.0, 1.0, 3.0];
        let group_ids = vec![0, 0, 1, 1];
        let input = vec![10.0, 20.0, 100.0, 200.0];

        let mut demeaner = SingleFEDemeaner::new(&weights, &group_ids, 2);
        let mut output = vec![0.0; 4];
        demeaner.demean(&input, &mut output);

        // Group 0: weighted_sum = 1*10 + 2*20 = 50, weight_sum = 3, mean = 50/3 ≈ 16.67
        // Group 1: weighted_sum = 1*100 + 3*200 = 700, weight_sum = 4, mean = 175.0
        let g0_mean = 50.0 / 3.0;
        let g1_mean = 700.0 / 4.0;

        assert!((output[0] - (10.0 - g0_mean)).abs() < 1e-10);
        assert!((output[1] - (20.0 - g0_mean)).abs() < 1e-10);
        assert!((output[2] - (100.0 - g1_mean)).abs() < 1e-10);
        assert!((output[3] - (200.0 - g1_mean)).abs() < 1e-10);
    }

    #[test]
    fn test_single_fe_inplace() {
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let group_ids = vec![0, 0, 1, 1];
        let mut data = vec![1.0, 3.0, 10.0, 20.0];

        let mut demeaner = SingleFEDemeaner::new(&weights, &group_ids, 2);
        demeaner.demean_inplace(&mut data);

        // Group 0 mean = 2.0, Group 1 mean = 15.0
        assert!((data[0] - (-1.0)).abs() < 1e-10);
        assert!((data[1] - 1.0).abs() < 1e-10);
        assert!((data[2] - (-5.0)).abs() < 1e-10);
        assert!((data[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_fe_single_group() {
        // All observations in one group - should all become zero-mean
        let weights = vec![1.0, 1.0, 1.0];
        let group_ids = vec![0, 0, 0];
        let input = vec![10.0, 20.0, 30.0];

        let mut demeaner = SingleFEDemeaner::new(&weights, &group_ids, 1);
        let mut output = vec![0.0; 3];
        demeaner.demean(&input, &mut output);

        // Mean = 20.0
        assert!((output[0] - (-10.0)).abs() < 1e-10);
        assert!((output[1] - 0.0).abs() < 1e-10);
        assert!((output[2] - 10.0).abs() < 1e-10);

        // Sum of demeaned values should be zero
        let sum: f64 = output.iter().sum();
        assert!(sum.abs() < 1e-10);
    }

    #[test]
    fn test_single_fe_many_groups() {
        // Each observation in its own group - demeaned values should all be zero
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let group_ids = vec![0, 1, 2, 3];
        let input = vec![10.0, 20.0, 30.0, 40.0];

        let mut demeaner = SingleFEDemeaner::new(&weights, &group_ids, 4);
        let mut output = vec![0.0; 4];
        demeaner.demean(&input, &mut output);

        // Each group has only one member, so mean = value, demeaned = 0
        for &v in &output {
            assert!(v.abs() < 1e-10);
        }
    }
}
