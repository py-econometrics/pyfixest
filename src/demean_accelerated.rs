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

struct FactorDemeaner {
    sample_weights: Vec<f64>,
    group_ids: Vec<usize>,
    group_weights: Vec<f64>,
    n_groups: usize,
    group_weighted_sums: Vec<f64>,
}

impl FactorDemeaner {
    fn new(
        sample_weights: Vec<f64>,
        group_ids: Vec<usize>,
        group_weights: Vec<f64>,
        n_groups: usize,
    ) -> Self {
        Self {
            sample_weights,
            group_ids,
            group_weights,
            n_groups,
            group_weighted_sums: vec![0.0; n_groups],
        }
    }

    fn project(&mut self, x: &mut [f64]) {
        self.group_weighted_sums.fill(0.0);

        // Accumulate weighted sums per group
        x.iter()
            .zip(&self.sample_weights)
            .zip(&self.group_ids)
            .for_each(|((&xi, &wi), &gid)| {
                self.group_weighted_sums[gid] += wi * xi;
            });

        // Compute group means and subtract from samples
        x.iter_mut()
            .zip(&self.group_ids)
            .for_each(|(xi, &gid)| {
                let group_mean = self.group_weighted_sums[gid] / self.group_weights[gid];
                *xi -= group_mean;
            });
    }
}

struct MultiFactorDemeaner {
    factors: Vec<FactorDemeaner>,
}

impl MultiFactorDemeaner {
    fn new(
        sample_weights: &[f64],
        group_ids: &[usize],
        group_weights: &[f64],
        n_samples: usize,
        n_factors: usize,
        n_groups: usize,
    ) -> Self {
        let mut factors = Vec::new();

        for j in 0..n_factors {
            // Extract group IDs for this factor
            let factor_group_ids: Vec<usize> = (0..n_samples)
                .map(|i| group_ids[i * n_factors + j])
                .collect();

            // Extract group weights for this factor
            let factor_group_weights = group_weights[j * n_groups..(j + 1) * n_groups].to_vec();

            factors.push(FactorDemeaner::new(
                sample_weights.to_vec(),
                factor_group_ids,
                factor_group_weights,
                n_groups,
            ));
        }

        Self { factors }
    }

    fn project(&mut self, x: &mut [f64]) {
        for factor in &mut self.factors {
            factor.project(x);
        }
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

    let process_column = |(k, mut col): (usize, ndarray::ArrayViewMut1<f64>)| {
        let mut xk_curr: Vec<f64> = (0..n_samples).map(|i| x[[i, k]]).collect();
        let mut xk_prev: Vec<f64> = xk_curr.iter().map(|&v| v - 1.0).collect();

        let mut demeaner = MultiFactorDemeaner::new(
            &sample_weights,
            &group_ids,
            &group_weights,
            n_samples,
            n_factors,
            n_groups,
        );

        let mut converged = false;
        for _ in 0..maxiter {
            demeaner.project(&mut xk_curr);

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
        py.allow_threads(|| demean_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter));

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
