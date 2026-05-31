use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    (0..x.ncols())
        .map(|col| x.column(col).iter().copied().collect())
        .collect()
}

fn demean_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<u32>,
    weights: Option<&ArrayView1<f64>>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    use_schwarz: bool,
) -> Result<(Array2<f64>, bool), within::WithinError> {
    let n_obs = x.nrows();
    let n_rhs = x.ncols();

    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|col| col.as_slice()).collect();
    let weights_vec: Option<Vec<f64>> = weights.map(|w| w.iter().copied().collect());
    let weights_ref = weights_vec.as_deref();

    let options = within::LsmrOptions {
        tol,
        maxiter,
        local_size,
    };

    let result = if use_schwarz {
        within::solve_batch(
            flist.view(),
            &x_slices,
            weights_ref,
            &options,
            None::<&within::PreconditionerConfig>,
        )?
    } else {
        let off = within::PreconditionerConfig::Off;
        within::solve_batch(
            flist.view(),
            &x_slices,
            weights_ref,
            &options,
            Some(&off),
        )?
    };

    let all_converged = result.converged.iter().all(|&c| c);
    let out = Array2::from_shape_vec((n_obs, n_rhs).f(), result.demeaned)
        .expect("within returns one demeaned column per RHS");
    Ok((out, all_converged))
}

#[pyfunction]
#[pyo3(signature = (
    x,
    flist,
    weights=None,
    tol=1e-8,
    maxiter=1_000,
    local_size=None,
    preconditioner="schwarz"
))]
pub fn _demean_within_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<u32>,
    weights: Option<PyReadonlyArray1<f64>>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    preconditioner: &str,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let use_schwarz = match preconditioner {
        "schwarz" => true,
        "none" => false,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "preconditioner={other:?} is not supported by the 'within' \
                 LSMR backend; use 'schwarz' (default) or 'none'."
            )));
        }
    };

    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_ref().map(|w| w.as_array());

    let (out, success) = py
        .detach(|| {
            demean_within_impl(
                &x_arr,
                &flist_arr,
                weights_arr.as_ref(),
                tol,
                maxiter,
                local_size,
                use_schwarz,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
