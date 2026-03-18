use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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

/// Convert a `usize` category array to `u32` in Fortran (column-major) order,
/// which is the layout `within::solve_batch` expects for best performance.
fn to_u32_fortran(flist: &ArrayView2<usize>) -> Array2<u32> {
    let (n_obs, n_factors) = flist.dim();
    let mut out = Array2::<u32>::zeros((n_obs, n_factors).f());
    for i in 0..n_obs {
        for j in 0..n_factors {
            out[[i, j]] = flist[[i, j]] as u32;
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
    let n_factors = flist.ncols();

    let categories = to_u32_fortran(flist);

    // Extract columns and build slice references for solve_batch
    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|c| c.as_slice()).collect();

    let weights_vec: Vec<f64> = weights.iter().cloned().collect();

    let params = within::SolverParams {
        tol,
        maxiter,
        ..within::SolverParams::default()
    };
    let preconditioner = within::Preconditioner::Additive(
        within::LocalSolverConfig::solver_default(),
        within::ReductionStrategy::Auto,
    );

    let result = within::solve_batch(
        categories.view(),
        &x_slices,
        Some(&weights_vec),
        &params,
        Some(&preconditioner),
    )?;

    // Assemble demeaned columns back into Array2
    let demeaned_cols: Vec<Vec<f64>> = (0..x_columns.len())
        .map(|k| result.demeaned(k).to_vec())
        .collect();
    let out = assemble_columns(&demeaned_cols, n_obs);

    let all_converged = result.converged().iter().all(|&c| c);
    Ok((out, all_converged))
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
