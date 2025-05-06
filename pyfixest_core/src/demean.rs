use numpy::{
    PyArray2,
    PyReadonlyArray2,
    PyReadonlyArray1,
};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
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
        group_weighted_sums.iter_mut().for_each(|s| *s = 0.0);
        for (i, &val) in x.iter().enumerate() {
            let id = group_ids[i];
            group_weighted_sums[id] += sample_weights[i] * val;
        }
        for (i, xi) in x.iter_mut().enumerate() {
            let id = group_ids[i];
            *xi -= group_weighted_sums[id] / group_weights[id];
        }
    }

    pub(super) fn calc_group_weights(
        sample_weights: &[f64],
        group_ids: &[usize],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Vec<f64> {
        let mut group_weights = vec![0.0; n_factors * n_groups];
        for j in 0..n_factors {
            for i in 0..n_samples {
                let id = group_ids[i * n_factors + j];
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
    let n_groups = flist.iter().cloned().max().unwrap() + 1;

    let sample_weights: Vec<f64> = weights.iter().cloned().collect();
    let group_ids: Vec<usize>        = flist.iter().cloned().collect();
    let group_weights = internal::calc_group_weights(
        &sample_weights, &group_ids, n_samples, n_factors, n_groups
    );

    let not_converged = Arc::new(AtomicUsize::new(0));

    let columns: Vec<Vec<f64>> = (0..n_features)
        .into_par_iter()
        .map(|k| {
            let mut xk_curr = vec![0.0; n_samples];
            let mut xk_prev = vec![0.0; n_samples];
            let mut gw_sums = vec![0.0; n_groups];

            for i in 0..n_samples {
                let v = x[[i, k]];
                xk_curr[i] = v;
                xk_prev[i] = v - 1.0;
            }

            let mut converged = false;
            for _ in 0..maxiter {
                for j in 0..n_factors {
                    let ids_j: Vec<usize> = (0..n_samples)
                        .map(|i| group_ids[i * n_factors + j])
                        .collect();
                    let gw_j = &group_weights[j * n_groups .. (j + 1) * n_groups];

                    internal::subtract_weighted_group_mean(
                        &mut xk_curr,
                        &sample_weights,
                        &ids_j,
                        gw_j,
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

            xk_curr
        })
        .collect();

    let mut res = Array2::<f64>::zeros((n_samples, n_features));
    for (k, col) in columns.into_iter().enumerate() {
        for i in 0..n_samples {
            res[[i, k]] = col[i];
        }
    }

    let success = not_converged.load(Ordering::SeqCst) == 0;
    (res, success)
}




#[pyfunction]
pub fn demean_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<usize>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_arr      = x.as_array();
    let flist_arr  = flist.as_array();
    let weights_arr= weights.as_array();

    let (out, success) = py.allow_threads(|| {
        demean_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter)
    });

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into_py(py), success))
}
