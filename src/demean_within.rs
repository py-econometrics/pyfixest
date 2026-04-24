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
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
    krylov: &str,
    preconditioner: &str,
    gmres_restart: usize,
) -> Result<(Array2<f64>, bool), within::WithinError> {
    let n_obs = x.nrows();
    let n_rhs = x.ncols();

    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|col| col.as_slice()).collect();
    let weights_vec: Vec<f64> = weights.iter().copied().collect();

    let krylov = match krylov {
        "cg" => within::KrylovMethod::Cg,
        "gmres" => within::KrylovMethod::Gmres {
            restart: gmres_restart,
        },
        _ => panic!("validated in Python: unsupported krylov method"),
    };
    let params = within::SolverParams {
        krylov,
        tol,
        maxiter,
        ..within::SolverParams::default()
    };
    let preconditioner = match preconditioner {
        "additive" => within::Preconditioner::Additive(
            within::LocalSolverConfig::solver_default(),
            within::ReductionStrategy::Auto,
        ),
        "multiplicative" => {
            within::Preconditioner::Multiplicative(within::LocalSolverConfig::solver_default())
        }
        _ => panic!("validated in Python: unsupported preconditioner"),
    };

    let result = within::solve_batch(
        flist.view(),
        &x_slices,
        Some(&weights_vec),
        &params,
        Some(&preconditioner),
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

    let (out, success) = py
        .detach(|| {
            demean_within_impl(
                &x_arr,
                &flist_arr,
                &weights_arr,
                tol,
                maxiter,
                krylov,
                preconditioner,
                gmres_restart,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
