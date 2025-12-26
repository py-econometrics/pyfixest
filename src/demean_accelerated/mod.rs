//! Accelerated alternating-projections demeaning with Irons-Tuck/Grand speedups.
//!
//! This module is a Rust port of fixest's original C++ demeaning implementation
//! (`https://github.com/lrberge/fixest/blob/master/src/demeaning.cpp`),
//! using coefficient-space iteration for efficiency.
//!
//! Dispatches based on number of fixed effects:
//! - 1 FE: O(n) closed-form solution (single pass, no iteration)
//! - 2 FE: Coefficient-space iteration with Irons-Tuck + Grand acceleration
//! - 3+ FE: Coefficient-space iteration with Irons-Tuck + Grand acceleration

mod coef_space;

use coef_space::{demean_single, FEInfo, FixestConfig};
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

    let config = FixestConfig {
        tol,
        maxiter,
        ..FixestConfig::default()
    };

    // Use the unified coefficient-space implementation for all FE counts
    demean_coef_space(
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

/// Demean using coefficient-space iteration (unified for all FE counts).
fn demean_coef_space(
    x: &ArrayView2<f64>,
    sample_weights: &[f64],
    group_ids: &[usize],
    n_samples: usize,
    n_features: usize,
    n_factors: usize,
    n_groups_per_factor: &[usize],
    config: &FixestConfig,
) -> (Array2<f64>, bool) {
    let not_converged = Arc::new(AtomicUsize::new(0));
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(k, mut col)| {
            let xk: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();

            let fe_info = FEInfo::new(
                n_samples,
                n_factors,
                group_ids,
                n_groups_per_factor,
                sample_weights,
            );

            let (result, _iter, converged) = demean_single(&fe_info, &xk, config);

            if !converged {
                not_converged.fetch_add(1, Ordering::SeqCst);
            }

            Zip::from(&mut col).and(&result).for_each(|col_elm, &val| {
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
