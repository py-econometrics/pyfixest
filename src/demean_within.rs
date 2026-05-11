use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn py_value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    (0..x.ncols())
        .map(|col| x.column(col).iter().copied().collect())
        .collect()
}

fn extract_krylov(krylov: &str, gmres_restart: usize) -> PyResult<within::KrylovMethod> {
    match krylov {
        "cg" => Ok(within::KrylovMethod::Cg),
        "gmres" => Ok(within::KrylovMethod::Gmres {
            restart: gmres_restart,
        }),
        _ => Err(py_value_error("`krylov` must be one of ('cg', 'gmres').")),
    }
}

fn extract_preconditioner(preconditioner: &str) -> PyResult<Option<within::Preconditioner>> {
    match preconditioner {
        "additive" => Ok(Some(within::Preconditioner::Additive(
            within::LocalSolverConfig::solver_default(),
            within::ReductionStrategy::Auto,
        ))),
        "multiplicative" => Ok(Some(within::Preconditioner::Multiplicative(
            within::LocalSolverConfig::solver_default(),
        ))),
        "off" => Ok(None),
        _ => Err(py_value_error(
            "`preconditioner` must be one of ('additive', 'multiplicative', 'off').",
        )),
    }
}

fn validate_cg_preconditioner(
    krylov: within::KrylovMethod,
    preconditioner: Option<&within::Preconditioner>,
) -> PyResult<()> {
    if matches!(krylov, within::KrylovMethod::Cg)
        && matches!(
            preconditioner,
            Some(within::Preconditioner::Multiplicative(_))
        )
    {
        return Err(py_value_error(
            "CG requires a symmetric preconditioner; use 'additive' or switch to GMRES",
        ));
    }
    Ok(())
}

fn demean_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<u32>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
    krylov: within::KrylovMethod,
    preconditioner: Option<&within::Preconditioner>,
) -> Result<(Array2<f64>, bool), within::WithinError> {
    let n_obs = x.nrows();
    let n_rhs = x.ncols();

    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|col| col.as_slice()).collect();
    let weights_vec: Vec<f64> = weights.iter().copied().collect();

    let params = within::SolverParams {
        krylov,
        tol,
        maxiter,
        ..within::SolverParams::default()
    };

    let result = within::solve_batch(
        flist.view(),
        &x_slices,
        Some(&weights_vec),
        &params,
        preconditioner,
    )?;

    let all_converged = result.converged().iter().all(|&c| c);
    let out = Array2::from_shape_vec((n_obs, n_rhs).f(), result.demeaned_all().to_vec())
        .expect("within returns one demeaned column per RHS");
    Ok((out, all_converged))
}

#[pyfunction]
#[pyo3(signature = (
    x,
    flist,
    weights,
    tol=1e-6,
    maxiter=1_000,
    krylov="cg",
    preconditioner="additive",
    gmres_restart=30
))]
pub fn _demean_within_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<u32>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
    krylov: &str,
    preconditioner: &str,
    gmres_restart: usize,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();
    let krylov = extract_krylov(krylov, gmres_restart)?;
    let preconditioner = extract_preconditioner(preconditioner)?;
    validate_cg_preconditioner(krylov, preconditioner.as_ref())?;

    let (out, success) = py
        .detach(|| {
            demean_within_impl(
                &x_arr,
                &flist_arr,
                &weights_arr,
                tol,
                maxiter,
                krylov,
                preconditioner.as_ref(),
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
