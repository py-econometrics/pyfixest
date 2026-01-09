//! Accelerated alternating-projections demeaning with Irons-Tuck/Grand speedups.
//!
//! This module is a Rust port of fixest's original C++ demeaning implementation
//! (`https://github.com/lrberge/fixest/blob/master/src/demeaning.cpp`),
//! using coefficient-space iteration for efficiency.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types
//!   - [`FixedEffectsIndex`](types::FixedEffectsIndex): Fixed effects indexing (which obs belongs to which group)
//!   - [`ObservationWeights`](types::ObservationWeights): Observation weights and group-level aggregations
//!   - [`DemeanContext`](DemeanContext): Combines index and weights for demeaning operations
//!   - [`FixestConfig`](FixestConfig): Algorithm parameters
//! - [`projection`]: Projection operations with [`Projector`](projection::Projector) trait
//!   - [`TwoFEProjector`](projection::TwoFEProjector): Specialized 2-FE projection
//!   - [`MultiFEProjector`](projection::MultiFEProjector): General Q-FE projection
//! - [`accelerator`]: Acceleration strategy
//!   - [`IronsTuckGrand`](accelerator::IronsTuckGrand): Irons-Tuck + Grand acceleration (matches fixest)
//! - [`demeaner`]: High-level solver strategies with [`Demeaner`](Demeaner) trait
//!   - [`SingleFEDemeaner`](SingleFEDemeaner): O(n) closed-form (1 FE)
//!   - [`TwoFEDemeaner`](TwoFEDemeaner): Accelerated iteration (2 FEs)
//!   - [`MultiFEDemeaner`](MultiFEDemeaner): Multi-phase strategy (3+ FEs)
//!
//! # Dispatching based on the number of fixed effects:
//! - 1 FE: O(n) closed-form solution (single pass, no iteration)
//! - 2 FE: Coefficient-space iteration with Irons-Tuck and Grand acceleration
//! - 3+ FE: Multi-phase strategy with 2-FE sub-convergence

pub mod accelerator;
pub mod demeaner;
pub mod projection;
pub mod types;

use demeaner::{Demeaner, MultiFEDemeaner, SingleFEDemeaner, TwoFEDemeaner};
use types::{ConvergenceState, DemeanContext, DemeanResult, FixestConfig};

use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;


/// Thread-local demeaner state that wraps the appropriate demeaner type.
///
/// This enum allows `for_each_init` to create a demeaner once per thread,
/// reusing its buffers across all columns processed by that thread.
enum ThreadLocalDemeaner<'a> {
    Single(SingleFEDemeaner<'a>),
    Two(TwoFEDemeaner<'a>),
    Multi(MultiFEDemeaner<'a>),
}

impl<'a> ThreadLocalDemeaner<'a> {
    /// Create a new thread-local demeaner based on the FE count.
    fn new(ctx: &'a DemeanContext, config: &'a FixestConfig) -> Self {
        match ctx.index.n_fe {
            1 => ThreadLocalDemeaner::Single(SingleFEDemeaner::new(ctx)),
            2 => ThreadLocalDemeaner::Two(TwoFEDemeaner::new(ctx, config)),
            _ => ThreadLocalDemeaner::Multi(MultiFEDemeaner::new(ctx, config)),
        }
    }

    /// Solve the demeaning problem, reusing internal buffers.
    #[inline]
    fn solve(&mut self, input: &[f64]) -> DemeanResult {
        match self {
            ThreadLocalDemeaner::Single(d) => d.solve(input),
            ThreadLocalDemeaner::Two(d) => d.solve(input),
            ThreadLocalDemeaner::Multi(d) => d.solve(input),
        }
    }
}

/// Result of batch demeaning operation.
pub(crate) struct DemeanBatchResult {
    pub demeaned: Array2<f64>,
    pub fe_coefficients: Array2<f64>,
    pub success: bool,
}

/// Demean using accelerated coefficient-space iteration.
///
/// Uses `for_each_init` to create one demeaner per thread, reusing buffers
/// across all columns processed by that thread.
///
/// # Returns
///
/// A `DemeanBatchResult` containing:
/// - `demeaned`: The demeaned data as an `Array2<f64>`
/// - `fe_coefficients`: FE coefficients as an `Array2<f64>`
/// - `success`: True if all columns converged
pub(crate) fn demean(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> DemeanBatchResult {
    let (n_samples, n_features) = x.dim();

    let config = FixestConfig {
        tol,
        maxiter,
        ..FixestConfig::default()
    };

    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut demeaned = Array2::<f64>::zeros((n_samples, n_features));

    // FEs are automatically reordered by size (largest first) for optimal convergence
    let ctx = DemeanContext::new(flist, weights);
    let n_coef = ctx.index.n_coef;

    let mut fe_coefficients = Array2::<f64>::zeros((n_coef, n_features));

    // Process columns in parallel, collecting both demeaned values and FE coefficients
    let results: Vec<(usize, DemeanResult)> = demeaned
        .axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .map_init(
            || ThreadLocalDemeaner::new(&ctx, &config),
            |demeaner, (k, _)| {
                let col_view = x.column(k);
                let result = if let Some(slice) = col_view.as_slice() {
                    demeaner.solve(slice)
                } else {
                    let xk: Vec<f64> = col_view.to_vec();
                    demeaner.solve(&xk)
                };
                (k, result)
            },
        )
        .collect();

    // Copy results back (sequential, but fast)
    for (k, result) in results {
        if result.convergence == ConvergenceState::NotConverged {
            not_converged.fetch_add(1, Ordering::SeqCst);
        }

        // Copy demeaned values
        for (i, &val) in result.demeaned.iter().enumerate() {
            demeaned[[i, k]] = val;
        }

        // Copy FE coefficients
        for (i, &val) in result.fe_coefficients.iter().enumerate() {
            fe_coefficients[[i, k]] = val;
        }
    }

    let success = not_converged.load(Ordering::SeqCst) == 0;
    DemeanBatchResult {
        demeaned,
        fe_coefficients,
        success,
    }
}

/// Python-exposed function for accelerated demeaning.
///
/// Returns a dict with:
/// - "demeaned": Array of demeaned values (n_samples, n_features)
/// - "fe_coefficients": Array of FE coefficients (n_coef, n_features)
/// - "success": Boolean indicating convergence
#[pyfunction]
#[pyo3(signature = (x, flist, weights, tol=1e-8, maxiter=100_000))]
pub fn _demean_rs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();

    let result = py.detach(|| demean(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let dict = PyDict::new(py);
    dict.set_item("demeaned", PyArray2::from_owned_array(py, result.demeaned))?;
    dict.set_item(
        "fe_coefficients",
        PyArray2::from_owned_array(py, result.fe_coefficients),
    )?;
    dict.set_item("success", result.success)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use demeaner::{MultiFEDemeaner, SingleFEDemeaner};
    use ndarray::{Array1, Array2};

    #[test]
    fn test_2fe_convergence() {
        let n_obs = 100;
        let n_fe = 2;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        let weights = Array1::<f64>::ones(n_obs);

        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged, "Should converge");
        assert!(result.iterations < 100, "Should converge quickly");
        assert!(result.demeaned.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_3fe_convergence() {
        let n_obs = 100;
        let n_fe = 3;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
            flist[[i, 2]] = i % 3;
        }

        let weights = Array1::<f64>::ones(n_obs);

        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = MultiFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged);
        assert!(result.demeaned.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_single_fe() {
        let n_obs = 100;
        let n_groups = 10;

        // Single fixed effect
        let mut flist = Array2::<usize>::zeros((n_obs, 1));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let mut demeaner = SingleFEDemeaner::new(&ctx);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged, "Single FE should always converge");
        assert_eq!(result.iterations, 0, "Single FE should be closed-form (0 iterations)");

        // Verify demeaning: each group's sum should be approximately 0
        for g in 0..n_groups {
            let group_sum: f64 = result
                .demeaned
                .iter()
                .enumerate()
                .filter(|(i, _)| i % n_groups == g)
                .map(|(_, &v)| v)
                .sum();
            assert!(
                group_sum.abs() < 1e-10,
                "Group {} sum should be ~0, got {}",
                g,
                group_sum
            );
        }
    }

    #[test]
    fn test_weighted_regression() {
        let n_obs = 100;
        let n_fe = 2;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        // Non-uniform weights: 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, ...
        let weights: Array1<f64> = (0..n_obs).map(|i| 1.0 + (i % 3) as f64).collect();
        let ctx = DemeanContext::new(&flist.view(), &weights.view());

        assert!(
            !ctx.weights.is_uniform,
            "Weights should be detected as non-uniform"
        );

        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();
        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged, "Weighted regression should converge");
        assert!(
            result.demeaned.iter().all(|&v| v.is_finite()),
            "All results should be finite"
        );
    }

    #[test]
    fn test_singleton_groups() {
        // Each observation in its own group for FE 0 (singleton groups)
        let n_obs = 20;

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i; // Singleton groups (each obs is its own group)
            flist[[i, 1]] = i % 4; // 4 groups in FE 1
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged, "Singleton groups should converge");

        // With singleton groups in FE 0, each observation's own mean is subtracted,
        // then adjusted for FE 1. The result should be all zeros since each
        // observation perfectly absorbs its own value in FE 0.
        assert!(
            result.demeaned.iter().all(|&v| v.abs() < 1e-10),
            "Singleton groups should yield near-zero residuals"
        );
    }

    #[test]
    fn test_small_groups() {
        // Test with very few observations per group
        let n_obs = 30;

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i / 3; // 10 groups, 3 obs each
            flist[[i, 1]] = i % 2; // 2 groups, 15 obs each
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged, "Small groups should converge");
        assert!(
            result.demeaned.iter().all(|&v| v.is_finite()),
            "All results should be finite"
        );
    }

    #[test]
    fn test_uniform_weights_detection() {
        let n_obs = 50;

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 5;
            flist[[i, 1]] = i % 3;
        }

        // Test uniform weights (all 1.0)
        let uniform_weights = Array1::<f64>::ones(n_obs);
        let ctx_uniform = DemeanContext::new(&flist.view(), &uniform_weights.view());
        assert!(
            ctx_uniform.weights.is_uniform,
            "All-ones weights should be detected as uniform"
        );

        // Test non-uniform weights
        let mut non_uniform_weights = Array1::<f64>::ones(n_obs);
        non_uniform_weights[0] = 2.0;
        let ctx_non_uniform = DemeanContext::new(&flist.view(), &non_uniform_weights.view());
        assert!(
            !ctx_non_uniform.weights.is_uniform,
            "Varying weights should be detected as non-uniform"
        );
    }

    #[test]
    fn test_buffer_reuse_produces_same_results() {
        // Test that solving multiple times with the same demeaner produces correct results
        let n_obs = 100;
        let n_fe = 2;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let config = FixestConfig::default();

        // Create a single demeaner and use it multiple times
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);

        let input1: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();
        let input2: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.2 + 1.0).collect();

        let result1a = demeaner.solve(&input1);
        let result2 = demeaner.solve(&input2);
        let result1b = demeaner.solve(&input1);

        // Results for the same input should be identical
        for (a, b) in result1a.demeaned.iter().zip(result1b.demeaned.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "Buffer reuse should produce identical results"
            );
        }

        // Results for different inputs should be different
        assert!(
            result1a
                .demeaned
                .iter()
                .zip(result2.demeaned.iter())
                .any(|(a, b)| (a - b).abs() > 0.01),
            "Different inputs should produce different results"
        );
    }

    // =========================================================================
    // FE Coefficient Tests
    // =========================================================================

    /// Helper: compute residuals by applying FE coefficients to observations.
    /// Returns input[i] - sum_q(coef[fe_q[i]]) for each observation.
    fn apply_coefficients(
        input: &[f64],
        flist: &Array2<usize>,
        fe_coefficients: &[f64],
        n_groups: &[usize],
    ) -> Vec<f64> {
        let n_obs = input.len();
        let n_fe = flist.ncols();

        // Compute coefficient offsets for each FE
        let mut coef_offsets = vec![0usize; n_fe];
        for q in 1..n_fe {
            coef_offsets[q] = coef_offsets[q - 1] + n_groups[q - 1];
        }

        (0..n_obs)
            .map(|i| {
                let mut fe_sum = 0.0;
                for q in 0..n_fe {
                    let g = flist[[i, q]];
                    fe_sum += fe_coefficients[coef_offsets[q] + g];
                }
                input[i] - fe_sum
            })
            .collect()
    }

    #[test]
    fn test_single_fe_coefficients() {
        let n_obs = 100;
        let n_groups = 10;

        let mut flist = Array2::<usize>::zeros((n_obs, 1));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let mut demeaner = SingleFEDemeaner::new(&ctx);
        let result = demeaner.solve(&input);

        // Verify coefficients are correct: applying them should give same residuals
        let reconstructed = apply_coefficients(&input, &flist, &result.fe_coefficients, &[n_groups]);

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-10,
                "Obs {}: demeaned ({}) != reconstructed ({})",
                i,
                demeaned,
                reconstructed
            );
        }

        // Verify coefficient count
        assert_eq!(
            result.fe_coefficients.len(),
            n_groups,
            "Should have {} coefficients",
            n_groups
        );
    }

    #[test]
    fn test_two_fe_coefficients_correct() {
        let n_obs = 100;
        let n_groups_0 = 10;
        let n_groups_1 = 5;

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups_0;
            flist[[i, 1]] = i % n_groups_1;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Verify coefficients are correct: applying them should give same residuals
        let reconstructed =
            apply_coefficients(&input, &flist, &result.fe_coefficients, &[n_groups_0, n_groups_1]);

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-8,
                "Obs {}: demeaned ({}) != reconstructed ({})",
                i,
                demeaned,
                reconstructed
            );
        }

        // Verify coefficient count
        assert_eq!(
            result.fe_coefficients.len(),
            n_groups_0 + n_groups_1,
            "Should have {} coefficients",
            n_groups_0 + n_groups_1
        );
    }

    #[test]
    fn test_two_fe_coefficients_ordering() {
        // Test that coefficients are returned in ORIGINAL FE order, not reordered
        let n_obs = 100;

        // FE 0: 5 groups (smaller), FE 1: 20 groups (larger)
        // Internally, FEs get reordered by size (largest first), so FE 1 becomes internal FE 0
        // But the coefficients should be returned in original order: [FE0 coeffs | FE1 coeffs]
        let n_groups_0 = 5; // smaller
        let n_groups_1 = 20; // larger

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups_0;
            flist[[i, 1]] = i % n_groups_1;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Verify coefficient count matches original ordering
        assert_eq!(
            result.fe_coefficients.len(),
            n_groups_0 + n_groups_1,
            "Should have {} coefficients",
            n_groups_0 + n_groups_1
        );

        // Verify coefficients are in original order by reconstructing residuals
        // using the ORIGINAL flist (not reordered)
        let reconstructed =
            apply_coefficients(&input, &flist, &result.fe_coefficients, &[n_groups_0, n_groups_1]);

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-8,
                "Obs {}: demeaned ({}) != reconstructed ({}) - coefficients may be in wrong order",
                i,
                demeaned,
                reconstructed
            );
        }
    }

    #[test]
    fn test_three_fe_coefficients_correct() {
        let n_obs = 120;
        let n_groups_0 = 10;
        let n_groups_1 = 6;
        let n_groups_2 = 4;

        let mut flist = Array2::<usize>::zeros((n_obs, 3));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups_0;
            flist[[i, 1]] = i % n_groups_1;
            flist[[i, 2]] = i % n_groups_2;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = MultiFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Verify coefficients are correct
        let reconstructed = apply_coefficients(
            &input,
            &flist,
            &result.fe_coefficients,
            &[n_groups_0, n_groups_1, n_groups_2],
        );

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-6,
                "Obs {}: demeaned ({}) != reconstructed ({})",
                i,
                demeaned,
                reconstructed
            );
        }

        // Verify coefficient count
        assert_eq!(
            result.fe_coefficients.len(),
            n_groups_0 + n_groups_1 + n_groups_2,
        );
    }

    #[test]
    fn test_three_fe_coefficients_ordering() {
        // Test that 3-FE coefficients are returned in original order
        let n_obs = 120;

        // Create FEs with different sizes to trigger reordering
        // Original: FE0=3 groups (smallest), FE1=15 groups (largest), FE2=8 groups (middle)
        // Reordered internally: FE1, FE2, FE0
        let n_groups_0 = 3; // smallest
        let n_groups_1 = 15; // largest
        let n_groups_2 = 8; // middle

        let mut flist = Array2::<usize>::zeros((n_obs, 3));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups_0;
            flist[[i, 1]] = i % n_groups_1;
            flist[[i, 2]] = i % n_groups_2;
        }

        let weights = Array1::<f64>::ones(n_obs);
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = MultiFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Verify coefficients work with ORIGINAL flist ordering
        let reconstructed = apply_coefficients(
            &input,
            &flist,
            &result.fe_coefficients,
            &[n_groups_0, n_groups_1, n_groups_2],
        );

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-6,
                "Obs {}: demeaned ({}) != reconstructed ({}) - coefficients may be in wrong order",
                i,
                demeaned,
                reconstructed
            );
        }
    }

    #[test]
    fn test_weighted_coefficients() {
        let n_obs = 100;
        let n_groups_0 = 10;
        let n_groups_1 = 5;

        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % n_groups_0;
            flist[[i, 1]] = i % n_groups_1;
        }

        // Non-uniform weights
        let weights: Array1<f64> = (0..n_obs).map(|i| 1.0 + (i % 3) as f64).collect();
        let ctx = DemeanContext::new(&flist.view(), &weights.view());
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Verify coefficients are correct with weighted reconstruction
        let reconstructed =
            apply_coefficients(&input, &flist, &result.fe_coefficients, &[n_groups_0, n_groups_1]);

        for (i, (&demeaned, &reconstructed)) in
            result.demeaned.iter().zip(reconstructed.iter()).enumerate()
        {
            assert!(
                (demeaned - reconstructed).abs() < 1e-8,
                "Weighted obs {}: demeaned ({}) != reconstructed ({})",
                i,
                demeaned,
                reconstructed
            );
        }
    }
}
