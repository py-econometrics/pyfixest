use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use numpy::IntoPyArray;

#[pyfunction]
pub fn find_collinear_variables_rs(
    py: Python,
    x: PyReadonlyArray2<f64>,
    tol: f64
) -> PyResult<(Py<PyArray2<bool>>, usize, bool)> {
    let x = x.as_array();
    let k = x.ncols();
    let mut r = ndarray::Array2::<f64>::zeros((k, k));
    let mut id_excl = vec![false; k];
    let mut n_excl = 0usize;
    let mut min_norm = x[(0,0)];

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
                let arr = ndarray::Array2::from_shape_vec((k, 1),
                    id_excl.iter().map(|&b| b).collect()
                ).unwrap();
                return Ok((arr.into_pyarray(py).to_owned(), n_excl, true));
            }
            continue;
        }

        if r_jj < min_norm {
            min_norm = r_jj;
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

    let arr = ndarray::Array2::from_shape_vec((k, 1),
        id_excl.iter().map(|&b| b).collect()
    ).unwrap();
    Ok((arr.into_pyarray(py).to_owned(), n_excl, false))
}
