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

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
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

/// Demean using accelerated coefficient-space iteration.
///
/// Uses `for_each_init` to create one demeaner per thread, reusing buffers
/// across all columns processed by that thread.
///
/// # Returns
///
/// A tuple of (demeaned_data, success) where:
/// - `demeaned_data`: The demeaned data as an `Array2<f64>`
/// - `success`: True if all columns converged
pub(crate) fn demean(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> (Array2<f64>, bool) {
    let (n_samples, n_features) = x.dim();

    let config = FixestConfig {
        tol,
        maxiter,
        ..FixestConfig::default()
    };

    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    // Use reorder_fe from config (default true, matching fixest)
    let ctx = DemeanContext::with_config(flist, weights, config.reorder_fe);

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each_init(
            // Init closure: called once per thread to create the thread-local state
            || ThreadLocalDemeaner::new(&ctx, &config),
            // Body closure: called for each column, reusing thread-local state
            |demeaner, (k, mut col)| {
                let col_view = x.column(k);
                // Zero-copy if the column is contiguous (F-order), otherwise copy
                let result = if let Some(slice) = col_view.as_slice() {
                    demeaner.solve(slice)
                } else {
                    let xk: Vec<f64> = col_view.to_vec();
                    demeaner.solve(&xk)
                };

                if result.convergence == ConvergenceState::NotConverged {
                    not_converged.fetch_add(1, Ordering::SeqCst);
                }

                Zip::from(&mut col)
                    .and(&result.demeaned)
                    .for_each(|col_elm, &val| {
                        *col_elm = val;
                    });
            },
        );

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}

/// Python-exposed function for accelerated demeaning.
///
/// Returns a tuple of (demeaned_array, success).
#[pyfunction]
#[pyo3(signature = (x, flist, weights, tol=1e-8, maxiter=100_000))]
pub fn _demean_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();

    let (demeaned, success) = py.detach(|| demean(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let pyarray = PyArray2::from_owned_array(py, demeaned);
    Ok((pyarray.into(), success))
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
}
