use numpy::{PyArray2, PyReadonlyArray2, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2, ArrayView1};
use numpy::IntoPyArray;

/// Detect multicollinear variables (Rust version of `_find_collinear_variables`).
#[pyfunction]
pub fn find_collinear_variables(
    py: Python,
    x: PyReadonlyArray2<f64>,
    tol: f64
) -> PyResult<(Py<PyArray2<bool>>, usize, bool)> {
    let X = x.as_array();
    let K = X.ncols();
    let mut R = ndarray::Array2::<f64>::zeros((K, K));
    let mut id_excl = vec![false; K];
    let mut n_excl = 0usize;
    let mut min_norm = X[(0,0)];

    for j in 0..K {
        // Compute R_jj = X[j,j] - Σ_{k<j & !id_excl[k]} R[k,j]²
        let mut R_jj = X[(j,j)];
        for k in 0..j {
            if id_excl[k] { continue; }
            let r_kj = R[(k,j)];
            R_jj -= r_kj * r_kj;
        }

        if R_jj < tol {
            id_excl[j] = true;
            n_excl += 1;
            if n_excl == K {
                // convert id_excl to a 2D bool array of shape (K,1)
                let arr = ndarray::Array2::from_shape_vec((K, 1),
                    id_excl.iter().map(|&b| b).collect()
                ).unwrap();
                return Ok((arr.into_pyarray(py).to_owned(), n_excl, true));
            }
            continue;
        }

        if R_jj < min_norm {
            min_norm = R_jj;
        }
        let Rjj_sqrt = R_jj.sqrt();
        R[(j,j)] = Rjj_sqrt;

        for i in (j+1)..K {
            let mut value = X[(i,j)];
            for k in 0..j {
                if id_excl[k] { continue; }
                value -= R[(k,i)] * R[(k,j)];
            }
            R[(j,i)] = value / Rjj_sqrt;
        }
    }

    // Build the final id_excl array (K×1) and return
    let arr = ndarray::Array2::from_shape_vec((K, 1),
        id_excl.iter().map(|&b| b).collect()
    ).unwrap();
    Ok((arr.into_pyarray(py).to_owned(), n_excl, false))
}
