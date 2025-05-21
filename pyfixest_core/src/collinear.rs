use ndarray::{Array1, Array2, ArrayView2};
use numpy::IntoPyArray;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

struct CollinearFoldState<'a> {
    x: ArrayView2<'a, f64>,
    r_matrix: Array2<f64>,
    collinear: Vec<bool>,
    n_collinear: usize,
}

impl<'a> CollinearFoldState<'a> {
    fn new(x: ArrayView2<'a, f64>) -> Self {
        let k = x.ncols();
        Self {
            x,
            r_matrix: Array2::<f64>::zeros((k, k)),
            collinear: vec![false; k],
            n_collinear: 0,
        }
    }

    fn mark_vector_as_collinear(&mut self, j: usize) {
        if !self.collinear[j] {
            self.collinear[j] = true;
            self.n_collinear += 1;
        }
    }

    fn update_r_matrix(&mut self, diag_val: f64, j: usize) {
        let k = self.x.ncols();
        let diag_val_sqrt = diag_val.sqrt();
        self.r_matrix[[j, j]] = diag_val_sqrt;

        let computed_row: Vec<(usize, f64)> = (j + 1..k)
            .into_par_iter()
            .map(|i| {
                let mut value = self.x[[i, j]];
                for p in 0..j {
                    if !self.collinear[p] {
                        value -= self.r_matrix[[p, i]] * self.r_matrix[[p, j]];
                    }
                }
                (i, value / diag_val_sqrt)
            })
            .collect();

        for (i, val) in computed_row {
            self.r_matrix[[j, i]] = val;
        }
    }

    fn compute_initial_diag_element(&mut self, j: usize) -> f64 {
        (0..j).fold(self.x[[j, j]], |mut r_jj, n| {
            if !self.collinear[n] {
                let r_kj = self.r_matrix[[n, j]];
                r_jj -= r_kj * r_kj;
            }
            r_jj
        })
    }

    fn into_result(self) -> (Array1<bool>, usize, bool) {
        let k = self.x.ncols();
        let collinear_array = Array1::from_vec(self.collinear);
        (collinear_array, self.n_collinear, self.n_collinear == k)
    }
}

#[pyfunction]
pub fn find_collinear_variables_rs(
    py: Python,
    x: PyReadonlyArray2<f64>,
    tol: f64,
) -> PyResult<(Py<PyArray1<bool>>, usize, bool)> {
    let x = x.as_array();
    let initial_state = CollinearFoldState::new(x);
    let final_state = (0..x.ncols()).fold(initial_state, |mut current_state, j| {
        let r_jj = current_state.compute_initial_diag_element(j);
        if r_jj < tol {
            current_state.mark_vector_as_collinear(j);
        } else {
            current_state.update_r_matrix(r_jj, j);
        }
        current_state
    });
    let (arr, n_excl, flag) = final_state.into_result();
    Ok((arr.into_pyarray(py).to_owned(), n_excl, flag))
}
