use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

mod internal {
    pub(super) fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    pub(super) fn subtract_weighted_group_mean(
        x: &mut [f64],
        sample_weights: &[f64],
        group_ids: &[usize],
        group_weights: &[f64],
        group_weighted_sums: &mut [f64],
    ) {
        group_weighted_sums.fill(0.0);

        // Accumulate weighted sums per group
        x.iter()
            .zip(sample_weights)
            .zip(group_ids)
            .for_each(|((&xi, &wi), &gid)| {
                group_weighted_sums[gid] += wi * xi;
            });

        // Compute group means
        let group_means: Vec<f64> = group_weighted_sums
            .iter()
            .zip(group_weights)
            .map(|(&sum, &weight)| sum / weight)
            .collect();

        // Subtract means from each sample
        x.iter_mut().zip(group_ids).for_each(|(xi, &gid)| {
            *xi -= group_means[gid];
        });
    }

    pub(super) fn calc_group_weights(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Vec<f64> {
        let mut group_weights = vec![0.0; n_factors * n_groups];
        for i in 0..n_samples {
            let weight = sample_weights[i];
            for j in 0..n_factors {
                let id = group_ids[i * n_factors + j];
                group_weights[j * n_groups + id] += weight;
            }
        }
        group_weights
    }
}

fn demean_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> (Array2<f64>, bool) {
    let (n_samples, n_features) = x.dim();
    let n_factors = flist.ncols();
    let n_groups = flist.iter().cloned().max().unwrap() + 1;

    let sample_weights: Vec<f64> = weights.iter().cloned().collect();
    let group_ids: Vec<usize> = flist.iter().cloned().collect();
    let group_weights =
        internal::calc_group_weights(&sample_weights, &group_ids, n_samples, n_factors, n_groups);

    let not_converged = Arc::new(AtomicUsize::new(0));

    // Precompute slices of group_ids for each factor
    let group_ids_by_factor: Vec<Vec<usize>> = (0..n_factors)
        .map(|j| {
            (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .collect()
        })
        .collect();

    // Precompute group weight slices
    let group_weight_slices: Vec<&[f64]> = (0..n_factors)
        .map(|j| &group_weights[j * n_groups..(j + 1) * n_groups])
        .collect();

    let process_column = |(k, mut col): (usize, ndarray::ArrayViewMut1<f64>)| {
        let mut xk_curr: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();
        let mut xk_prev: Vec<f64> = xk_curr.iter().map(|&v| v - 1.0).collect();
        let mut gw_sums = vec![0.0; n_groups];

        let mut converged = false;
        for _ in 0..maxiter {
            for j in 0..n_factors {
                internal::subtract_weighted_group_mean(
                    &mut xk_curr,
                    &sample_weights,
                    &group_ids_by_factor[j],
                    group_weight_slices[j],
                    &mut gw_sums,
                );
            }

            if internal::sad_converged(&xk_curr, &xk_prev, tol) {
                converged = true;
                break;
            }
            xk_prev.copy_from_slice(&xk_curr);
        }

        if !converged {
            not_converged.fetch_add(1, Ordering::SeqCst);
        }
        Zip::from(&mut col).and(&xk_curr).for_each(|col_elm, &val| {
            *col_elm = val;
        });
    };

    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    res.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(process_column);

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}


/// Demean a 2D array x by a set of fixed effects using the alternating
/// projection algorithm.
///
/// Parameters
/// ----------
/// x : np.ndarray[float64]
///     2D array of data to be demeaned (shape: observations x variables).
/// flist : np.ndarray[usize]
///     2D array of group indicators (shape: observations x the number of fixed effects), must be integer-encoded.
/// weights : np.ndarray[float64]
///     1D array of observation weights (length: observations).
/// tol : float, optional
///     Convergence tolerance (default: 1e-8).
/// maxiter : int, optional
///     Maximum number of iterations (default: 100000).
///
/// Returns
/// -------
/// (np.ndarray[float64], bool)
///     Tuple with:
///         - demeaned array (same shape as `x`)
///         - success flag (True if converged, False if maxiter was reached)
///
/// Notes
/// -----
/// This function performs iterative demeaning to remove all group means specified by
/// `flist` from the data `x`, optionally using observation weights. Convergence is
/// determined when the change between iterations falls below `tol`.
/// Note that flist must be a 2D array of integers. NaNs are not allowed in
/// either `x` or `flist`.
///
/// Example
/// -------
/// ```python
/// import numpy as np
/// from pyfixest.core.demean import _demean_rs
///
/// # Sample data: 5 observations, 2 variables
/// x = np.array([[10.0, 2.0],
///               [11.0, 3.0],
///               [12.0, 4.0],
///               [20.0, 5.0],
///               [21.0, 6.0]])
///
/// # Grouping by two categorical variables, integer-encoded
/// flist = np.array([[0, 1],
///                   [0, 2],
///                   [0, 2],
///                   [1, 1],
///                   [1, 2]])
///
/// # All observations equally weighted
/// weights = np.ones(5)
///
/// # Call the function
/// x_demeaned, converged = _demean_rs(x, flist, weights)
///
/// print("Demeaned x:")
/// print(x_demeaned)
/// print("Converged:", converged)
/// ```

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

    let (out, success) = py.detach(|| demean_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
