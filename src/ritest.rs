use faer::prelude::*;
use faer::Side;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::demean::demean_impl;

fn ndarray_to_faer(m: &ArrayView2<f64>) -> faer::Mat<f64> {
    let (rows, cols) = m.dim();
    faer::Mat::<f64>::from_fn(rows, cols, |i, j| m[[i, j]])
}

fn ndarray_vec_to_faer(v: &[f64]) -> faer::Mat<f64> {
    faer::Mat::<f64>::from_fn(v.len(), 1, |i, _| v[i])
}

struct FwlProjector {
    llt: faer::linalg::solvers::Llt<f64>,
    x_f: faer::Mat<f64>,
    y_tilde: Array1<f64>,
}

fn run_ri_impl(
    resampled_d: &ArrayView2<f64>,
    y_demean: &ArrayView1<f64>,
    x_demean2: &ArrayView2<f64>,
    fval: Option<&ArrayView2<usize>>,
    weights: &ArrayView1<f64>,
) -> Array1<f64> {
    let (n, reps) = resampled_d.dim();
    let k = x_demean2.ncols();

    let projector = if k > 0 {
        let x_f = ndarray_to_faer(x_demean2);
        let xtx = x_f.transpose() * &x_f;
        let llt = xtx.llt(Side::Lower).expect("X'X is not positive definite");

        let y_col = ndarray_vec_to_faer(y_demean.as_slice().unwrap());
        let xty = x_f.transpose() * &y_col;
        let beta_y = llt.solve(&xty);
        let y_hat = &x_f * &beta_y;

        let yt = Array1::from_iter((0..n).map(|i| y_demean[i] - y_hat[(i, 0)]));

        Some(FwlProjector { llt, x_f, y_tilde: yt })
    } else {
        None
    };

    let y_tilde_ref = projector.as_ref().map_or(y_demean.to_owned(), |p| p.y_tilde.clone());

    let ri_coefs: Vec<f64> = (0..reps)
        .into_par_iter()
        .map(|i| {
            let d_col = resampled_d.column(i).to_owned();

            let d_demeaned = if let Some(fv) = fval {
                let d_2d = d_col.insert_axis(Axis(1));
                let (demeaned, _) = demean_impl(&d_2d.view(), fv, weights, 1e-8, 100_000);
                demeaned.column(0).to_owned()
            } else {
                d_col
            };

            let d_tilde = if let Some(ref proj) = projector {
                let d_f = ndarray_vec_to_faer(d_demeaned.as_slice().unwrap());
                let xtd = proj.x_f.transpose() * &d_f;
                let beta_d = proj.llt.solve(&xtd);
                let d_hat = &proj.x_f * &beta_d;
                Array1::from_iter(
                    (0..d_demeaned.len()).map(|j| d_demeaned[j] - d_hat[(j, 0)]),
                )
            } else {
                d_demeaned
            };

            d_tilde.dot(&y_tilde_ref) / d_tilde.dot(&d_tilde)
        })
        .collect();

    Array1::from(ri_coefs)
}

#[pyfunction]
pub fn _run_ri_rs(
    py: Python<'_>,
    resampled_d: PyReadonlyArray2<f64>,
    y_demean: PyReadonlyArray1<f64>,
    x_demean2: PyReadonlyArray2<f64>,
    fval: Option<PyReadonlyArray2<usize>>,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let resampled_d_arr = resampled_d.as_array();
    let y_arr = y_demean.as_array();
    let x_arr = x_demean2.as_array();
    let w_arr = weights.as_array();

    let fval_arr = fval.as_ref().map(|f| f.as_array());

    let result = py.allow_threads(|| {
        run_ri_impl(
            &resampled_d_arr,
            &y_arr,
            &x_arr,
            fval_arr.as_ref(),
            &w_arr,
        )
    });

    let pyarray = PyArray1::from_owned_array(py, result);
    Ok(pyarray.into())
}
