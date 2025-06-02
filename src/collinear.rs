use ndarray::{Array1, Array2, ArrayView2};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use thiserror::Error;


#[derive(Debug, Error)]
enum CollinearityError {
    #[error("Input matrix must be square, got {rows}x{cols}")]
    NonSquareMatrix { rows: usize, cols: usize },

    #[error("Tolerance must be positive and finite, got {value}")]
    InvalidTolerance { value: f64 },
}


/// Detect collinear (linearly dependent) columns in a symmetric matrix.
///
/// Parameters
/// ----------
/// x : ndarray-like of shape (p, p), dtype float64
///     Symmetric (Gram) matrix `X.T @ X`.
/// tol : float
///     Multicollinearity threshold.
///
/// Returns
/// -------
/// mask : ndarray of bool, shape (p,)
///     Boolean indicator of collinear columns
/// n_excl : int
///     Number of columns flagged as collinear
/// all_collinear : bool
///     `True` if all columns are collinear.
///
/// * `x` - Input matrix (must be square, typically X'X in a regression model, where X is the N x k design matrix)
/// * `tol` - Tolerance for detecting collinearity (smaller values require closer to exact linear dependence)
///
/// Notes
/// -----
///
/// The detection order depends on the original column ordering; an
/// order-independent variant would add **column pivoting** (choose, at each
/// step, the remaining column with the largest residual variance).

fn find_collinear_variables_impl(
    x: ArrayView2<f64>,
    tol: f64,
) -> Result<(Array1<bool>, usize, bool), CollinearityError> {
    // Validate tolerance
    if tol <= 0.0 {
        return Err(CollinearityError::InvalidTolerance { value: tol });
    }


    let k = x.ncols();
    if !x.is_square() {
        return Err(CollinearityError::NonSquareMatrix {rows: x.nrows(), cols: k})
    }

    let mut r = Array2::<f64>::zeros((k, k));
    let mut id_excl = vec![false; k];
    let mut n_excl = 0usize;

    for j in 0..k {
        let mut r_jj = x[(j,j)];
        for k in 0..j {
            if id_excl[k] { continue; }
            let r_kj = r[(k,j)];
            r_jj -= r_kj * r_kj;
        }

        if r_jj < tol {
            id_excl[j] = true;
            n_excl += 1;
            if n_excl == k {
                let arr = Array1::from_vec(id_excl);
                return Ok((arr, n_excl, true));
            }
            continue;
        }

        let rjj_sqrt = r_jj.sqrt();
        r[(j,j)] = rjj_sqrt;

        for i in (j+1)..k {
            let mut value = x[(i,j)];
            for k in 0..j {
                if id_excl[k] { continue; }
                value -= r[(k,i)] * r[(k,j)];
            }
            r[(j,i)] = value / rjj_sqrt;
        }
    }

    let arr = Array1::from_vec(id_excl);
    Ok((arr, n_excl, false))
}

/// Detect collinear (linearly dependent) columns in a square matrix.
///
/// Uses a Cholesky-based algorithm to identify variables (columns) that are collinear or nearly collinear,
/// based on a user-specified tolerance.
///
/// Parameters
/// ----------
/// x : numpy.ndarray (float64)
///     A square 2D array (n x n) whose columns will be checked for collinearity.
/// tol : float, optional
///     Threshold below which a variable is considered collinear (default is 1e-10).
///
/// Returns
/// -------
/// mask : numpy.ndarray (bool)
///     Boolean array of length `n`. `True` indicates that the column is collinear and should be excluded.
/// n_excluded : int
///     Number of columns detected as collinear.
/// all_collinear : bool
///     `True` if all columns are collinear (e.g., zero or singular matrix), else `False`.
///
/// Raises
/// ------
/// ValueError
///     If the input matrix is not square, or if the tolerance is not positive.
///
/// Notes: This function is a translation of Laurent Bergé's c++ implementation in
/// the fixest package.

#[pyfunction]
#[pyo3(signature = (x, tol=1e-10))]
pub fn _find_collinear_variables_rs(
    py: Python,
    x: PyReadonlyArray2<f64>,
    tol: f64,
) -> PyResult<(Py<PyArray1<bool>>, usize, bool)> {
    let x = x.as_array();
    // Call the implementation and convert any errors to Python ValueError
    match find_collinear_variables_impl(x, tol) {
        Ok((arr, n_excl, flag)) => Ok((arr.into_pyarray(py).to_owned(), n_excl, flag)),
        Err(err) => {
            // Convert Rust errors to Python ValueError
            Err(PyValueError::new_err(err.to_string()))
        }
    }
}
