//! Accelerated alternating-projections demeaning with Irons-Tuck/Grand speedups.
//!
//! This module is a Rust port of fixest's original C++ demeaning implementation
//! (`https://github.com/lrberge/fixest/blob/master/src/demeaning.cpp`),
//! using coefficient-space iteration for efficiency.
//!
//! # Module Structure
//!
//! - [`types`]: Core data types
//!   - [`Dimensions`](types::Dimensions): Problem shape
//!   - [`Weights`](types::Weights): Observation weights
//!   - [`FixedEffectInfo`](types::FixedEffectInfo): Per-FE information
//!   - [`DemeanContext`](DemeanContext): Combines all context for demeaning
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
mod sweep;
pub mod types;

use demeaner::{Demeaner, MultiFEDemeaner, SingleFEDemeaner, TwoFEDemeaner};
use types::{ConvergenceState, DemeanContext, DemeanMultiResult, DemeanResult, FixestConfig};

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
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
    #[inline]
    fn new(ctx: &'a DemeanContext, config: &'a FixestConfig) -> Self {
        match ctx.dims.n_fe {
            1 => ThreadLocalDemeaner::Single(SingleFEDemeaner::new(ctx)),
            2 => ThreadLocalDemeaner::Two(TwoFEDemeaner::new(ctx, config)),
            _ => ThreadLocalDemeaner::Multi(MultiFEDemeaner::new(ctx, config)),
        }
    }

    /// Solve the demeaning problem, reusing internal buffers.
    #[inline(always)]
    fn solve(&mut self, input: &[f64]) -> DemeanResult {
        match self {
            ThreadLocalDemeaner::Single(d) => d.solve(input),
            ThreadLocalDemeaner::Two(d) => d.solve(input),
            ThreadLocalDemeaner::Multi(d) => d.solve(input),
        }
    }
}

/// Demean using accelerated coefficient-space iteration.
///
/// Uses `for_each_init` to create one demeaner per thread, reusing buffers
/// across all columns processed by that thread.
///
/// # Arguments
///
/// * `x` - Input data array (n_samples, n_features)
/// * `flist` - Fixed effect group IDs (n_samples, n_fe)
/// * `weights` - Per-observation weights, or None for unweighted
/// * `tol` - Convergence tolerance
/// * `maxiter` - Maximum iterations
///
/// # Returns
///
/// A [`DemeanMultiResult`] containing demeaned data, FE coefficients, and convergence status.
pub(crate) fn demean(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: Option<&ArrayView1<f64>>,
    tol: f64,
    maxiter: usize,
) -> DemeanMultiResult {
    let (n_samples, n_features) = x.dim();

    let config = FixestConfig {
        tol,
        maxiter,
        ..FixestConfig::default()
    };

    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut demeaned = Array2::<f64>::zeros((n_samples, n_features));

    // Create context (FEs are always reordered by size, matching fixest)
    let ctx = DemeanContext::new(flist, weights);
    let n_coef = ctx.dims.n_coef;

    let mut fe_coefficients = Array2::<f64>::zeros((n_coef, n_features));

    // Process columns in parallel, collecting both demeaned values and FE coefficients
    demeaned
        .axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .zip(
            fe_coefficients
                .axis_iter_mut(ndarray::Axis(1))
                .into_par_iter(),
        )
        .enumerate()
        .for_each_init(
            // Init closure: called once per thread to create the thread-local state
            || ThreadLocalDemeaner::new(&ctx, &config),
            // Body closure: called for each column, reusing thread-local state
            |demeaner, (k, (mut dem_col, mut coef_col))| {
                let col_view = x.column(k);
                // Zero-copy if the column is contiguous (F-order), otherwise copy
                let result = if let Some(slice) = col_view.as_slice() {
                    demeaner.solve(slice)
                } else {
                    let xk: Vec<f64> = col_view.to_vec();
                    demeaner.solve(&xk)
                };

                if result.convergence == ConvergenceState::NotConverged {
                    not_converged.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }

                Zip::from(&mut dem_col)
                    .and(&result.demeaned)
                    .for_each(|col_elm, &val| {
                        *col_elm = val;
                    });

                Zip::from(&mut coef_col)
                    .and(&result.fe_coefficients)
                    .for_each(|col_elm, &val| {
                        *col_elm = val;
                    });
            },
        );

    let success = not_converged.load(std::sync::atomic::Ordering::Relaxed) == 0;
    DemeanMultiResult {
        demeaned,
        fe_coefficients,
        success,
    }
}

/// Python-exposed function for accelerated demeaning.
///
/// # Arguments
///
/// * `x` - Input data array (n_samples, n_features)
/// * `flist` - Fixed effect group IDs (n_samples, n_fe)
/// * `weights` - Per-observation weights, or None for unweighted (fast path)
/// * `tol` - Convergence tolerance (default: 1e-8)
/// * `maxiter` - Maximum iterations (default: 100_000)
///
/// # Returns
///
/// A dict with:
/// - "demeaned": Array of demeaned values (n_samples, n_features)
/// - "fe_coefficients": Array of FE coefficients (n_coef, n_features)
/// - "success": Boolean indicating convergence
#[pyfunction]
#[pyo3(signature = (x, flist, weights=None, tol=1e-8, maxiter=100_000))]
pub fn _demean_rs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: Option<PyReadonlyArray1<f64>>,
    tol: f64,
    maxiter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_ref().map(|w| w.as_array());

    let result = py.detach(|| demean(&x_arr, &flist_arr, weights_arr.as_ref(), tol, maxiter));

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
    use demeaner::MultiFEDemeaner;
    use ndarray::Array2;

    #[test]
    fn test_2fe_convergence() {
        let n_obs = 100;
        let n_fe = 2;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        // Unweighted case
        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(
            result.convergence,
            ConvergenceState::Converged,
            "Should converge"
        );
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

        // Unweighted case
        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = MultiFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged);
        assert!(result.demeaned.iter().all(|&v| v.is_finite()));

        // Verify demeaning: each FE group's sum should be approximately 0
        let group_counts = [10, 5, 3];
        for q in 0..n_fe {
            for g in 0..group_counts[q] {
                let group_sum: f64 = result
                    .demeaned
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| flist[[*i, q]] == g)
                    .map(|(_, &v)| v)
                    .sum();
                assert!(
                    group_sum.abs() < 1e-8,
                    "FE {} group {} sum should be ~0, got {}",
                    q,
                    g,
                    group_sum
                );
            }
        }
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

        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let mut demeaner = SingleFEDemeaner::new(&ctx);
        let result = demeaner.solve(&input);

        assert_eq!(
            result.convergence,
            ConvergenceState::Converged,
            "Single FE should always converge"
        );
        assert_eq!(
            result.iterations, 0,
            "Single FE should be closed-form (0 iterations)"
        );

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
        let weights: ndarray::Array1<f64> = (0..n_obs).map(|i| 1.0 + (i % 3) as f64).collect();
        let ctx =
            DemeanContext::new(&flist.view(), Some(&weights.view()));

        assert!(
            ctx.weights.is_some(),
            "Weights should be Some when provided"
        );

        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();
        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(
            result.convergence,
            ConvergenceState::Converged,
            "Weighted regression should converge"
        );
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

        // Test with no weights (None) - unweighted case
        let ctx_unweighted = DemeanContext::new(&flist.view(), None);
        assert!(
            ctx_unweighted.weights.is_none(),
            "No weights should result in weights=None"
        );

        // Test with weights (Some) - weighted case
        let weights: ndarray::Array1<f64> = (0..n_obs).map(|i| 1.0 + (i % 2) as f64).collect();
        let ctx_weighted =
            DemeanContext::new(&flist.view(), Some(&weights.view()));
        assert!(
            ctx_weighted.weights.is_some(),
            "Provided weights should result in weights=Some"
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

        let ctx = DemeanContext::new(&flist.view(), None);
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
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_single_observation() {
        // Edge case: only 1 observation
        let flist = Array2::<usize>::zeros((1, 2));
        let ctx = DemeanContext::new(&flist.view(), None);

        let input = vec![42.0];
        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged);
        // With a single observation, demeaned value should be 0 (input - mean = 0)
        assert!(
            result.demeaned[0].abs() < 1e-10,
            "Single observation should demean to 0"
        );
    }

    #[test]
    fn test_single_group_per_fe() {
        // Edge case: all observations in the same group for each FE
        let n_obs = 50;
        let flist = Array2::<usize>::zeros((n_obs, 2)); // All zeros = single group each

        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged);
        // All in same group means demeaned = input - mean(input)
        let mean: f64 = input.iter().sum::<f64>() / n_obs as f64;
        for (i, &val) in result.demeaned.iter().enumerate() {
            let expected = input[i] - mean;
            assert!(
                (val - expected).abs() < 1e-10,
                "Demeaned value should equal input - mean"
            );
        }
    }

    #[test]
    fn test_many_groups() {
        // Edge case: many groups (each observation in its own group for FE0)
        let n_obs = 200;
        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i; // Each obs in its own group
            flist[[i, 1]] = i % 5;
        }

        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(result.convergence, ConvergenceState::Converged);
        assert!(result.demeaned.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_extreme_weight_ratios() {
        // Edge case: very different weights
        let n_obs = 100;
        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        // Extreme weights: 0.001, 1000, 0.001, 1000, ...
        let weights: ndarray::Array1<f64> = (0..n_obs)
            .map(|i| if i % 2 == 0 { 0.001 } else { 1000.0 })
            .collect();

        let ctx =
            DemeanContext::new(&flist.view(), Some(&weights.view()));
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig::default();
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        assert_eq!(
            result.convergence,
            ConvergenceState::Converged,
            "Should converge even with extreme weight ratios"
        );
        assert!(
            result.demeaned.iter().all(|&v| v.is_finite()),
            "All results should be finite"
        );
    }

    // =========================================================================
    // Convergence Failure Tests
    // =========================================================================

    #[test]
    fn test_small_maxiter_produces_valid_results() {
        // Test that even with very limited iterations, results are valid (finite)
        // The accelerated algorithm may still converge quickly for simple problems
        let n_obs = 100;
        let n_fe = 2;

        let mut flist = Array2::<usize>::zeros((n_obs, n_fe));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 10;
            flist[[i, 1]] = i % 5;
        }

        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        // Use maxiter=1 - algorithm may or may not converge depending on data
        let config = FixestConfig {
            maxiter: 1,
            ..FixestConfig::default()
        };
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // Regardless of convergence, results should be finite
        assert!(
            result.demeaned.iter().all(|&v| v.is_finite()),
            "Results should be finite even with limited iterations"
        );
        assert!(
            result.iterations <= 1,
            "Should have at most 1 iteration"
        );
    }

    #[test]
    fn test_convergence_failure_with_zero_maxiter() {
        // Edge case: maxiter=0
        let n_obs = 50;
        let mut flist = Array2::<usize>::zeros((n_obs, 2));
        for i in 0..n_obs {
            flist[[i, 0]] = i % 5;
            flist[[i, 1]] = i % 3;
        }

        let ctx = DemeanContext::new(&flist.view(), None);
        let input: Vec<f64> = (0..n_obs).map(|i| (i as f64) * 0.1).collect();

        let config = FixestConfig {
            maxiter: 0,
            ..FixestConfig::default()
        };
        let mut demeaner = TwoFEDemeaner::new(&ctx, &config);
        let result = demeaner.solve(&input);

        // With maxiter=0, should not converge (unless already converged after init)
        // The exact behavior depends on implementation, but results should be finite
        assert!(result.demeaned.iter().all(|&v| v.is_finite()));
    }
}
