use ndarray::{Array1, Array2, ArrayView2};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
enum CollinearityError {
    #[error("Input matrix must be square, got {rows}x{cols}")]
    NonSquareMatrix { rows: usize, cols: usize },

    #[error("Tolerance must be positive and finite, got {value}")]
    InvalidTolerance { value: f64 },
}

/// State struct for the collinearity detection algorithm using a modified Cholesky decomposition.
///
/// This struct holds all the necessary data during the algorithm's execution:
/// - The input matrix reference
/// - The R matrix for Cholesky factorization
/// - Tracking of which columns are collinear
/// - Count of collinear columns found
struct CollinearFoldState<'a> {
    x: ArrayView2<'a, f64>,
    r_matrix: Array2<f64>,
    collinear: Vec<bool>,
    n_collinear: usize,
}

impl<'a> CollinearFoldState<'a> {
    /// Creates a new state for the collinearity detection algorithm.
    ///
    /// # Arguments
    ///
    /// * `x` - Input matrix view (must be square, typically a correlation or covariance matrix)
    ///
    /// # Returns
    ///
    /// A new `CollinearFoldState` with initialized values:
    /// - R matrix filled with zeros
    /// - Empty collinearity tracking
    /// - Zero collinear count
    fn new(x: ArrayView2<'a, f64>) -> Result<Self, CollinearityError> {
        let n_cols = x.ncols();
        let n_rows = x.nrows();

        if !x.is_square() {
            return Err(CollinearityError::NonSquareMatrix {
                rows: n_rows,
                cols: n_cols,
            });
        }
        // Validate that the matrix is square
        Ok(Self {
            x,
            r_matrix: Array2::<f64>::zeros((n_cols, n_cols)),
            collinear: vec![false; n_cols],
            n_collinear: 0,
        })
    }

    /// Marks a column as collinear and updates the collinear count.
    ///
    /// This method is idempotent - calling it multiple times on the same
    /// column will only increment the counter once.
    ///
    /// # Arguments
    ///
    /// * `j` - Index of the column to mark as collinear
    #[inline]
    fn mark_vector_as_collinear(&mut self, j: usize) {
        if !self.collinear[j] {
            self.collinear[j] = true;
            self.n_collinear += 1;
        }
    }

    /// Updates the R matrix for a non-collinear column using parallel computation.
    ///
    /// This is the core computational step of the algorithm. For column j, it:
    /// 1. Stores the square root of the diagonal element
    /// 2. Computes the off-diagonal elements in parallel using Rayon
    /// 3. Updates the R matrix with the computed values
    ///
    /// # Arguments
    ///
    /// * `diag_val` - Computed diagonal value (r_jj) for column j
    /// * `j` - Index of the column being processed
    fn update_r_matrix(&mut self, diag_val: f64, j: usize) {
        let k = self.x.ncols();
        let diag_val_sqrt = diag_val.sqrt();

        // Store the square root on the diagonal
        self.r_matrix[[j, j]] = diag_val_sqrt;

        // Parallel computation of the off-diagonal elements
        let computed_row: Vec<(usize, f64)> = (j + 1..k)
            .into_par_iter()
            .map(|i| {
                // Start with the value from the input matrix
                let mut value = self.x[[i, j]];
                for p in 0..j {
                    // Subtract contributions from previous non-collinear columns
                    if !self.collinear[p] {
                        value -= self.r_matrix[[p, i]] * self.r_matrix[[p, j]];
                    }
                }
                (i, value / diag_val_sqrt)
            })
            .collect();

        // Update the R matrix with the computed values
        for (i, val) in computed_row {
            self.r_matrix[[j, i]] = val;
        }
    }

    /// Computes the diagonal element r_jj for column j.
    ///
    /// This method calculates the diagonal element after removing the contributions
    /// from all previous non-collinear columns. A small value indicates collinearity.
    ///
    /// # Arguments
    ///
    /// * `j` - Index of the column to compute the diagonal element for
    ///
    /// # Returns
    ///
    /// The computed diagonal element value
    #[inline]
    fn compute_initial_diag_element(&mut self, j: usize) -> f64 {
        // Start with the diagonal element from the input matrix
        (0..j).fold(self.x[[j, j]], |mut r_jj, n| {
            // Skip collinear columns
            if !self.collinear[n] {
                let r_nj = self.r_matrix[[n, j]];
                r_jj -= r_nj * r_nj;
            }
            r_jj
        })
    }

    /// Converts the state into the final result tuple.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - An ndarray Array1 indicating which columns are collinear
    /// - Number of collinear columns found
    /// - Boolean flag indicating if all columns are collinear
    fn into_result(self) -> (Array1<bool>, usize, bool) {
        let k = self.x.ncols();
        let collinear_array = Array1::from_vec(self.collinear);
        (collinear_array, self.n_collinear, self.n_collinear == k)
    }
}

/// Detects collinear variables in a matrix using a modified Cholesky decomposition.
///
/// This algorithm identifies which columns in the input matrix can be expressed
/// as linear combinations of other columns, within the specified tolerance.
///
/// # Arguments
///
/// * `x` - Input matrix (must be square, typically a correlation or covariance matrix)
/// * `tol` - Tolerance for detecting collinearity (smaller values require closer to exact linear dependence)
///
/// # Returns
///
/// A tuple containing:
/// - Boolean array indicating which columns are collinear
/// - Number of collinear columns found
/// - Flag indicating if all columns are collinear
fn find_collinear_variables_impl(
    x: ArrayView2<f64>,
    tol: f64,
) -> Result<(Array1<bool>, usize, bool), CollinearityError> {
    // Validate tolerance
    if tol <= 0.0 {
        return Err(CollinearityError::InvalidTolerance { value: tol });
    }

    // Create initial state
    let initial_state = CollinearFoldState::new(x)?;

    // Process each column sequentially using fold
    // The algorithm is inherently sequential as each column depends on previous ones
    let final_state = (0..x.ncols()).fold(initial_state, |mut current_state, j| {
        // Compute the diagonal element after projections onto previous columns
        let r_jj = current_state.compute_initial_diag_element(j);
        // Check for collinearity using the specified tolerance
        if r_jj < tol {
            // If the diagonal element is too small, mark the column as collinear
            current_state.mark_vector_as_collinear(j);
        } else {
            // Otherwise, update the R matrix for this non-collinear column
            current_state.update_r_matrix(r_jj, j);
        }
        // Return the updated state for the next iteration
        current_state
    });
    Ok(final_state.into_result())
}

/// Detects collinear variables in a matrix using a modified Cholesky decomposition.
///
/// This algorithm identifies which columns in the input matrix can be expressed
/// as linear combinations of other columns, within the specified tolerance.
/// It's particularly useful for identifying multicollinearity in statistical models.
///
/// # Arguments
///
/// * `py` - Python interpreter token
/// * `x` - Input matrix (must be square, typically a correlation or covariance matrix)
/// * `tol` - Tolerance for detecting collinearity (smaller values require closer to exact linear dependence)
///
/// # Returns
///
/// A PyResult containing a tuple with:
/// - Boolean array indicating which columns are collinear
/// - Number of collinear columns found
#[pyfunction]
pub fn find_collinear_variables_rs(
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
