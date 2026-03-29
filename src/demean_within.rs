use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass(module = "pyfixest.core._core_impl", name = "_WithinPreconditionerHandle")]
#[derive(Clone)]
pub struct WithinPreconditionerHandle {
    inner: within::FePreconditioner,
}

impl WithinPreconditionerHandle {
    fn from_inner(inner: within::FePreconditioner) -> Self {
        Self { inner }
    }
}

fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    (0..x.ncols())
        .map(|col| x.column(col).iter().copied().collect())
        .collect()
}

fn build_solver_params(
    tol: f64,
    maxiter: usize,
    krylov_method: &str,
    gmres_restart: usize,
) -> PyResult<within::SolverParams> {
    let krylov = match krylov_method {
        "cg" => within::KrylovMethod::Cg,
        "gmres" => within::KrylovMethod::Gmres {
            restart: gmres_restart,
        },
        _ => {
            return Err(PyValueError::new_err(
                "krylov_method must be either 'cg' or 'gmres'.",
            ));
        }
    };

    Ok(within::SolverParams {
        tol,
        maxiter,
        krylov,
        ..within::SolverParams::default()
    })
}

fn build_preconditioner_config(preconditioner_type: &str) -> PyResult<within::Preconditioner> {
    match preconditioner_type {
        "additive" => Ok(within::Preconditioner::Additive(
            within::LocalSolverConfig::solver_default(),
            within::ReductionStrategy::Auto,
        )),
        "multiplicative" => Ok(within::Preconditioner::Multiplicative(
            within::LocalSolverConfig::solver_default(),
        )),
        _ => Err(PyValueError::new_err(
            "preconditioner_type must be either 'additive' or 'multiplicative'.",
        )),
    }
}

fn build_preconditioner_solver_params(preconditioner_type: &str) -> within::SolverParams {
    match preconditioner_type {
        "multiplicative" => within::SolverParams {
            krylov: within::KrylovMethod::Gmres { restart: 30 },
            ..within::SolverParams::default()
        },
        _ => within::SolverParams::default(),
    }
}

fn build_weighted_design(
    flist: &ArrayView2<u32>,
    weights: &ArrayView1<f64>,
) -> PyResult<within::WeightedDesign<within::FactorMajorStore>> {
    let n_obs = flist.nrows();
    let n_factors = flist.ncols();
    let factor_levels: Vec<Vec<u32>> = (0..n_factors)
        .map(|factor| flist.column(factor).iter().copied().collect())
        .collect();
    let weights_vec: Vec<f64> = weights.iter().copied().collect();
    let store = within::FactorMajorStore::new(
        factor_levels,
        within::ObservationWeights::Dense(weights_vec),
        n_obs,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    within::WeightedDesign::from_store(store).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

fn solve_result_to_array(
    result: within::BatchSolveResult,
    n_obs: usize,
    n_rhs: usize,
) -> (Array2<f64>, bool) {
    let all_converged = result.converged().iter().all(|&c| c);
    let out = Array2::from_shape_vec((n_obs, n_rhs).f(), result.demeaned_all().to_vec())
        .expect("within returns one demeaned column per RHS");
    (out, all_converged)
}

fn solve_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<u32>,
    weights: &ArrayView1<f64>,
    tol: f64,
    maxiter: usize,
    krylov_method: &str,
    gmres_restart: usize,
    preconditioner_type: &str,
    preconditioner_handle: Option<&WithinPreconditionerHandle>,
) -> PyResult<(Array2<f64>, bool)> {
    let params = build_solver_params(tol, maxiter, krylov_method, gmres_restart)?;
    let n_obs = x.nrows();
    let n_rhs = x.ncols();
    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|col| col.as_slice()).collect();

    let result = if let Some(handle) = preconditioner_handle {
        let design = build_weighted_design(flist, weights)?;
        let solver = within::Solver::from_design_with_preconditioner(
            design,
            &params,
            handle.inner.clone(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        solver
            .solve_batch(&x_slices)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        let preconditioner = build_preconditioner_config(preconditioner_type)?;
        let design = build_weighted_design(flist, weights)?;
        let solver = within::Solver::from_design(design, &params, Some(&preconditioner))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        solver
            .solve_batch(&x_slices)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };

    Ok(solve_result_to_array(result, n_obs, n_rhs))
}

fn build_preconditioner_impl(
    flist: &ArrayView2<u32>,
    weights: &ArrayView1<f64>,
    preconditioner_type: &str,
) -> PyResult<WithinPreconditionerHandle> {
    let preconditioner_config = build_preconditioner_config(preconditioner_type)?;
    let design = build_weighted_design(flist, weights)?;
    let params = build_preconditioner_solver_params(preconditioner_type);
    let solver = within::Solver::from_design(design, &params, Some(&preconditioner_config))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let built = solver
        .preconditioner()
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err("solver did not build a preconditioner"))?;
    Ok(WithinPreconditionerHandle::from_inner(built))
}

#[pyfunction]
#[pyo3(signature = (flist, weights, preconditioner_type="additive"))]
pub fn _build_within_preconditioner_rs(
    py: Python<'_>,
    flist: PyReadonlyArray2<u32>,
    weights: PyReadonlyArray1<f64>,
    preconditioner_type: &str,
) -> PyResult<Py<WithinPreconditionerHandle>> {
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();
    let handle =
        py.detach(|| build_preconditioner_impl(&flist_arr, &weights_arr, preconditioner_type))?;
    Py::new(py, handle)
}

#[pyfunction]
#[pyo3(signature = (
    x,
    flist,
    weights,
    tol=1e-6,
    maxiter=1_000,
    krylov_method="cg",
    gmres_restart=30,
    preconditioner_type="additive",
    preconditioner_handle=None
))]
pub fn _demean_within_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<u32>,
    weights: PyReadonlyArray1<f64>,
    tol: f64,
    maxiter: usize,
    krylov_method: &str,
    gmres_restart: usize,
    preconditioner_type: &str,
    preconditioner_handle: Option<Py<WithinPreconditionerHandle>>,
) -> PyResult<(Py<PyArray2<f64>>, bool)> {
    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_array();
    let preconditioner_handle = preconditioner_handle
        .as_ref()
        .map(|handle| handle.bind(py).borrow().clone());

    let (out, success) = py.detach(|| {
        solve_within_impl(
            &x_arr,
            &flist_arr,
            &weights_arr,
            tol,
            maxiter,
            krylov_method,
            gmres_restart,
            preconditioner_type,
            preconditioner_handle.as_ref(),
        )
    })?;

    let pyarray = PyArray2::from_owned_array(py, out);
    Ok((pyarray.into(), success))
}
