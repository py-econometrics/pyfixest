use numpy::{PyArray2, PyReadonlyArray2, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2, ArrayView1};
use std::sync::atomic::{AtomicUsize, Ordering};
use numpy::IntoPyArray;


// Internal helper functions
mod internal {
    /// Check if all absolute differences between `a` and `b` are below `tol`.
    pub(super) fn sad_converged(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    /// Subtracts the weighted group mean from `x` in place.
    pub(super) fn subtract_weighted_group_mean(
        x: &mut [f64],
        sample_weights: &[f64],
        group_ids: &[usize],
        group_weights: &[f64],
        group_weighted_sums: &mut [f64],
    ) {
        // Zero accumulators
        group_weighted_sums.iter_mut().for_each(|s| *s = 0.0);
        // Compute weighted sums per group
        for (i, &val) in x.iter().enumerate() {
            let id = group_ids[i];
            group_weighted_sums[id] += sample_weights[i] * val;
        }
        // Subtract group mean
        for (i, xi) in x.iter_mut().enumerate() {
            let id = group_ids[i];
            *xi -= group_weighted_sums[id] / group_weights[id];
        }
    }

    /// Compute total weights per group and factor.
    /// Compute total weights per group *and* factor in factor‑major order.
    pub(super) fn calc_group_weights(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Vec<f64> {
        // layout = [ factor0: [g0, g1, …], factor1: [g0, g1, …], … ]
        let mut group_weights = vec![0.0; n_factors * n_groups];
        for j in 0..n_factors {
            for i in 0..n_samples {
                let id = group_ids[i * n_factors + j];
                // now contiguous block per j
                group_weights[j * n_groups + id] += sample_weights[i];
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

    // Determine number of clusters (groups)
    let n_groups = flist.iter().cloned().max().unwrap() + 1;

    // Force contiguous buffers instead of unwrap():
    let sample_weights_vec: Vec<f64> = weights.iter().cloned().collect();
    let sample_weights: &[f64] = &sample_weights_vec;

    let group_ids_vec: Vec<usize> = flist.iter().cloned().collect();
    let group_ids: &[usize] = &group_ids_vec;

    // Precompute group_weights: length = n_groups * n_factors
    let group_weights = internal::calc_group_weights(
        sample_weights,
        group_ids,
        n_samples,
        n_factors,
        n_groups,
    );

    // Atomic counter for features that did not converge
    let not_converged = AtomicUsize::new(0);

    // Prepare output array
    let mut res = Array2::<f64>::zeros((n_samples, n_features));

    for k in 0..n_features {
        let mut xk_curr = vec![0.0; n_samples];
        let mut xk_prev = vec![0.0; n_samples];
        let mut group_weighted_sums = vec![0.0; n_groups];

        // Initialize
        for i in 0..n_samples {
            let val = x[[i, k]];
            xk_curr[i] = val;
            xk_prev[i] = val - 1.0;
        }

        // Alternating projections
        let mut converged = false;
        for _ in 0..maxiter {
            for j in 0..n_factors {
                // Extract IDs for factor j
                let group_col_ids: Vec<usize> = (0..n_samples)
                    .map(|i| group_ids[i * n_factors + j])
                    .collect();
                let gw_slice = &group_weights[j * n_groups..(j + 1) * n_groups];

                internal::subtract_weighted_group_mean(
                    &mut xk_curr,
                    sample_weights,
                    &group_col_ids,
                    gw_slice,
                    &mut group_weighted_sums,
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

        // Write result
        for i in 0..n_samples {
            res[[i, k]] = xk_curr[i];
        }
    }

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}


#[pyfunction]
pub fn demean(
    py: Python,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    // Convert NumPy arrays to ndarray views
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();

    // Call the internal Rust function
    let (demeaned, success) = demean_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter);

    // Convert result back to NumPy array
    let out = PyArray2::from_owned_array(py, demeaned);
    Ok((out.into_py(py), success))
}

/// Module definition
#[pymodule]
fn mymodule(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(demean, m)?)?;
    Ok(())
}
