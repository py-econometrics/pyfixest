use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;


#[allow(non_snake_case)]
pub fn demean_impl(X: &mut Array2<f64>, D: ArrayView2<usize>, weights: ArrayView1<f64>, tol: f64, iterations: usize) -> bool {
    let nsamples = X.nrows();
    let nfactors = D.ncols();
    let success = Arc::new(AtomicBool::new(true));
    let group_weights = FactorGroupWeights::new(&D, &weights);

    X.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .for_each(|mut column| {
            let mut demeaner = ColumnDemeaner::new(nsamples, group_weights.width);

            for _ in 0..iterations {
                for i in 0..nfactors {
                    demeaner.demean_column(
                        &mut column,
                        &weights,
                        &D.column(i),
                        group_weights.factor_weight_slice(i)
                    );
                }

                demeaner.check_convergence(&column.view(), tol);
                if demeaner.converged {
                    break;
                }
            }

            if !demeaner.converged {
                // We can use a relaxed ordering since we only ever go from true to false
                // and it doesn't matter how many times we do this.
                success.store(false, Ordering::Relaxed);
            }
        });

    success.load(Ordering::Relaxed)
}

// The column demeaner is in charge of subtracting group means until convergence.
struct ColumnDemeaner {
    converged: bool,
    checkpoint: Vec<f64>,
    group_sums: Vec<f64>,
}

impl ColumnDemeaner {
    fn new(n: usize, k: usize) -> Self {
        Self {
            converged: false,
            checkpoint: vec![0.0; n],
            group_sums: vec![0.0; k],
        }
    }

    fn demean_column(
        &mut self,
        x: &mut ArrayViewMut1<f64>,
        weights: &ArrayView1<f64>,
        groups: &ArrayView1<usize>,
        group_weights: &[f64],
    ) {
        self.group_sums.fill(0.0);

        // Compute group sums
        for ((&xi, &wi), &gid) in x.iter().zip(weights).zip(groups) {
            self.group_sums[gid] += wi * xi;
        }

        // Convert sums to means
        self.group_sums
            .iter_mut()
            .zip(group_weights.iter())
            .for_each(|(sum, &weight)| {
                *sum /= weight
            });

        // Subtract group means
        for (xi, &gid) in x.iter_mut().zip(groups) {
            *xi -= self.group_sums[gid] // Really these are means now
        }
    }


    // Check elementwise convergence and update checkpoint
    fn check_convergence(
        &mut self,
        x: &ArrayView1<f64>,
        tol: f64,
    ) {
        self.converged = true; // Innocent until proven guilty
        x.iter()
            .zip(self.checkpoint.iter_mut())
            .for_each(|(&xi, cp)| {
                if (xi - *cp).abs() > tol {
                    self.converged = false; // Guilty!
                }
                *cp = xi; // Update checkpoint
            });
    }
}

// Instead of recomputing the denominators for the weighted group averages every time,
// we'll precompute them and store them in a grid-like structure. The grid will have
// dimensions (m, k) where m is the number of factors and k is the maximum group ID.
struct FactorGroupWeights {
    values: Vec<f64>,
    width: usize,
}

impl FactorGroupWeights {
    fn new(flist: &ArrayView2<usize>, weights: &ArrayView1<f64>) -> Self {
        let n_samples = flist.nrows();
        let n_factors = flist.ncols();
        let width = flist.iter().max().unwrap() + 1;

        let mut values = vec![0.0; n_factors * width];
        for i in 0..n_samples {
            let weight = weights[i];
            for j in 0..n_factors {
                let id = flist[[i, j]];
                values[j * width + id] += weight;
            }
        }

        Self {
            values,
            width,
        }
    }

    fn factor_weight_slice(&self, factor_index: usize) -> &[f64] {
        &self.values[factor_index * self.width..(factor_index + 1) * self.width]
    }
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
    let mut x_array = x.as_array().to_owned();
    let flist_array = flist.as_array();
    let weights_array = weights.as_array();

    let converged = demean_impl(
        &mut x_array,
        flist_array,
        weights_array,
        tol,
        maxiter,
    );

    let pyarray = PyArray2::from_owned_array(py, x_array);
    Ok((pyarray.into_py(py), converged))
}
