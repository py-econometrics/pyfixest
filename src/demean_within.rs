use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Convert pyfixest's row-major `(n_obs x n_factors)` layout to within's
/// factor-major `Vec<Vec<i64>>`, and compute `n_levels` per factor (max+1).
fn transpose_flist(flist: &ArrayView2<usize>) -> (Vec<Vec<i64>>, Vec<usize>) {
    let n_factors = flist.ncols();
    let n_obs = flist.nrows();
    let mut categories: Vec<Vec<i64>> = vec![vec![0i64; n_obs]; n_factors];
    let mut n_levels: Vec<usize> = vec![0; n_factors];

    for j in 0..n_factors {
        let col = &mut categories[j];
        let mut max_val: usize = 0;
        for i in 0..n_obs {
            let v = flist[[i, j]];
            col[i] = v as i64;
            if v > max_val {
                max_val = v;
            }
        }
        n_levels[j] = max_val + 1;
    }
    (categories, n_levels)
}

/// Extract columns from a 2D array view as Vec<Vec<f64>>.
fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    let (n_obs, n_cols) = x.dim();
    (0..n_cols)
        .map(|k| (0..n_obs).map(|i| x[[i, k]]).collect())
        .collect()
}

/// Assemble column vectors back into an Array2.
fn assemble_columns(columns: &[Vec<f64>], n_obs: usize) -> Array2<f64> {
    let n_cols = columns.len();
    let mut out = Array2::<f64>::zeros((n_obs, n_cols));
    for (k, col) in columns.iter().enumerate() {
        for i in 0..n_obs {
            out[[i, k]] = col[i];
        }
    }
    out
}

fn demean_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<usize>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
) -> Result<(Array2<f64>, bool), within::WithinError> {
    let n_obs = x.nrows();

    let (categories, n_levels) = transpose_flist(flist);
    let weights_vec: Vec<f64> = weights.iter().cloned().collect();
    let columns = extract_columns(x);

    let result = within::demean_batch_default(
        categories, n_levels, n_obs, weights_vec, &columns, tol, maxiter,
    )?;

    Ok((assemble_columns(&result.columns, n_obs), result.all_converged))
}

#[pyfunction]
#[pyo3(signature = (x, flist, weights, tol=1e-6, maxiter=1_000))]
pub fn _demean_within_rs(
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

    let (out, success) = py
        .detach(|| demean_within_impl(&x_arr, &flist_arr, &weights_arr, tol, maxiter))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
