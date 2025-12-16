//! Optimized demeaning implementation with Irons-Tuck acceleration.
//!
//! Dispatches based on number of fixed effects:
//! - 1 FE: O(n) closed-form solution (single pass, no iteration)
//! - 2 FE: Alternating projections with Irons-Tuck acceleration
//! - 3+ FE: Alternating projections with Irons-Tuck + Grand acceleration

pub mod acceleration;
pub mod buffers;
pub mod general;
pub mod simd_ops;
pub mod single_fe;
pub mod two_fe;

use acceleration::{GrandAcceleration, IronsTuckAcceleration, Projector, StepResult};
use general::MultiFactorProjector;
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use two_fe::TwoFEDemeaner;

/// Configuration for accelerated demeaning.
struct AcceleratedConfig {
    tol: f64,
    maxiter: usize,
    /// Warmup iterations before checking convergence
    warmup_iters: usize,
    /// Interval between Irons-Tuck acceleration steps
    accel_interval: usize,
    /// Interval for grand acceleration snapshots (3+ FEs only)
    grand_accel_interval: usize,
    /// Interval for SSR-based convergence safeguard
    ssr_check_interval: usize,
    /// Iteration from which to run a post-acceleration projection (fixest: 3)
    post_accel_iter: usize,
}

impl AcceleratedConfig {
    fn new(tol: f64, maxiter: usize) -> Self {
        Self {
            tol,
            maxiter,
            // Match fixest defaults: warmup 15, accel every 3, grand every 15.
            warmup_iters: 15,
            accel_interval: 3,
            grand_accel_interval: 15,
            // fixest uses a coarse SSR check every 40 iterations.
            ssr_check_interval: 40,
            // fixest projects once after acceleration starting at iter 3.
            post_accel_iter: 3,
        }
    }
}

/// Accelerated alternating-projections demeaning with Irons-Tuck/Grand speedups.
pub(crate) fn demean_accelerated(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> (Array2<f64>, bool) {
    let (n_samples, n_features) = x.dim();
    let n_factors = flist.ncols();

    let sample_weights: Vec<f64> = weights.iter().cloned().collect();
    let group_ids: Vec<usize> = flist.iter().cloned().collect();

    // Special case: single FE uses O(n) closed-form solution
    if n_factors == 1 {
        return demean_single_fe(x, &sample_weights, &group_ids, n_samples, n_features);
    }

    // Compute n_groups per factor
    let n_groups_per_factor: Vec<usize> = (0..n_factors)
        .map(|j| {
            (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .max()
                .unwrap_or(0)
                + 1
        })
        .collect();

    let config = AcceleratedConfig::new(tol, maxiter);

    // 2 FE case: Use TwoFEDemeaner with Irons-Tuck acceleration
    if n_factors == 2 {
        return demean_two_fe(
            x,
            &sample_weights,
            &group_ids,
            n_samples,
            n_features,
            n_factors,
            &n_groups_per_factor,
            &config,
        );
    }

    // 3+ FE case: Use MultiFactorProjector with Irons-Tuck + Grand acceleration
    demean_multi_fe(
        x,
        &sample_weights,
        &group_ids,
        n_samples,
        n_features,
        n_factors,
        &n_groups_per_factor,
        &config,
    )
}

/// Demean with single fixed effect (O(n) closed-form solution).
fn demean_single_fe(
    x: &ArrayView2<f64>,
    sample_weights: &[f64],
    group_ids: &[usize],
    n_samples: usize,
    n_features: usize,
) -> (Array2<f64>, bool) {
    let n_groups = group_ids.iter().cloned().max().unwrap() + 1;
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(k, mut col)| {
            let xk: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();
            let mut output = vec![0.0; n_samples];

            let mut demeaner =
                single_fe::SingleFEDemeaner::new(sample_weights, group_ids, n_groups);
            demeaner.demean(&xk, &mut output);

            Zip::from(&mut col)
                .and(&output)
                .for_each(|col_elm, &val| {
                    *col_elm = val;
                });
        });

    (res, true)
}

/// Demean with two fixed effects using Irons-Tuck acceleration.
fn demean_two_fe(
    x: &ArrayView2<f64>,
    sample_weights: &[f64],
    group_ids: &[usize],
    n_samples: usize,
    n_features: usize,
    n_factors: usize,
    n_groups_per_factor: &[usize],
    config: &AcceleratedConfig,
) -> (Array2<f64>, bool) {
    // Extract group IDs for each factor
    let fe1_ids: Vec<usize> = (0..n_samples)
        .map(|i| group_ids[i * n_factors])
        .collect();
    let fe2_ids: Vec<usize> = (0..n_samples)
        .map(|i| group_ids[i * n_factors + 1])
        .collect();

    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(k, mut col)| {
            let xk: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();

            let projector = TwoFEDemeaner::new(
                sample_weights,
                &fe1_ids,
                &fe2_ids,
                n_groups_per_factor[0],
                n_groups_per_factor[1],
            );

            let mut accel = IronsTuckAcceleration::new(projector, n_samples);
            accel.set_initial(&xk);

            let mut converged = false;
            for iter in 0..config.maxiter {
                // Apply acceleration every accel_interval iterations after warmup
                let should_accelerate =
                    iter >= config.warmup_iters && iter % config.accel_interval == 0;

                let step_result = accel.step(should_accelerate);

                if step_result == StepResult::NumericallyConverged {
                    converged = true;
                    break;
                }

                // Check convergence after warmup
                if iter >= config.warmup_iters && accel.is_converged(config.tol) {
                    converged = true;
                    break;
                }
            }

            if !converged {
                not_converged.fetch_add(1, Ordering::SeqCst);
            }

            let result = accel.get_result();
            Zip::from(&mut col).and(result).for_each(|col_elm, &val| {
                *col_elm = val;
            });
        });

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}

/// Demean with 3+ fixed effects using Irons-Tuck + Grand acceleration.
fn demean_multi_fe(
    x: &ArrayView2<f64>,
    sample_weights: &[f64],
    group_ids: &[usize],
    n_samples: usize,
    n_features: usize,
    n_factors: usize,
    n_groups_per_factor: &[usize],
    config: &AcceleratedConfig,
) -> (Array2<f64>, bool) {
    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(k, mut col)| {
            let xk = x.column(k).to_vec();
            let original = xk.clone();
            let mut prev_ssr = f64::MAX;

            let projector = MultiFactorProjector::new(
                sample_weights,
                group_ids,
                n_samples,
                n_factors,
                n_groups_per_factor,
            );

            let mut accel = IronsTuckAcceleration::new(projector, n_samples);
            let mut grand_accel = GrandAcceleration::new(n_samples, config.grand_accel_interval);

            accel.set_initial(&xk);

            let mut converged = false;
            for iter in 0..config.maxiter {
                // Apply Irons-Tuck acceleration every accel_interval iterations after warmup
                let should_accelerate =
                    iter >= config.warmup_iters && iter % config.accel_interval == 0;

                let step_result = accel.step(should_accelerate);

                if step_result == StepResult::NumericallyConverged {
                    converged = true;
                    break;
                }

                // After acceleration, fixest runs an extra projection (iter_projAfterAcc).
                if iter >= config.post_accel_iter {
                    let _ = accel.regular_step();
                }

                // Grand acceleration: record snapshots and apply when ready
                if grand_accel.should_record(iter) {
                    grand_accel.record(accel.get_result());

                    if grand_accel.can_apply() {
                        // Apply grand acceleration
                        let current = accel.get_result().to_vec();
                        let mut accelerated = current;
                        grand_accel.apply(&mut accelerated);
                        accel.set_initial(&accelerated);
                    }
                }

                // Check convergence after warmup
                if iter >= config.warmup_iters && accel.is_converged(config.tol) {
                    converged = true;
                    break;
                }

                // Coarse SSR check to stop slow tails (matches fixest safeguard).
                if iter > 0 && iter % config.ssr_check_interval == 0 {
                    let current = accel.get_result();
                    let mut ssr = 0.0;
                    for i in 0..n_samples {
                        let r = original[i] - current[i];
                        ssr += r * r;
                    }
                    let rel_change = (prev_ssr - ssr).abs() / (prev_ssr.abs() + 1e-12);
                    if rel_change < config.tol {
                        converged = true;
                        break;
                    }
                    prev_ssr = ssr;
                }
            }

            if !converged {
                not_converged.fetch_add(1, Ordering::SeqCst);
            }

            let result = accel.get_result();
            Zip::from(&mut col).and(result).for_each(|col_elm, &val| {
                *col_elm = val;
            });
        });

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}

#[pyfunction]
#[pyo3(signature = (x, flist, weights, tol=1e-8, maxiter=100_000))]
pub fn _demean_accelerated_rs(
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

    let (out, success) =
        py.detach(|| demean_accelerated(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
