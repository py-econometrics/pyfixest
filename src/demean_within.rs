use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

/// Opaque handle to a pre-built within preconditioner (Additive Schwarz or
/// Diagonal Jacobi).
///
/// Equality / hashing follow Python's pyo3 defaults (object identity), in
/// line with upstream ``within._within.Preconditioner``. Pickle uses
/// ``postcard`` round-tripping via ``__reduce__``.
#[pyclass(frozen, name = "Preconditioner", module = "pyfixest.core._core_impl")]
pub struct PyPreconditioner {
    inner: within::Preconditioner,
}

#[pymethods]
impl PyPreconditioner {
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        let inner: within::Preconditioner = postcard::from_bytes(data).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to deserialize preconditioner: {}",
                e
            ))
        })?;
        Ok(Self { inner })
    }

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }

    #[getter]
    fn variant(&self) -> &'static str {
        self.inner.variant_name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Preconditioner(variant={}, nrows={}, ncols={})",
            self.inner.variant_name(),
            self.inner.nrows(),
            self.inner.ncols()
        )
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = postcard::to_stdvec(&self.inner).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to serialize preconditioner: {}",
                e
            ))
        })?;
        let cls = py.get_type::<Self>();
        let py_bytes = PyBytes::new(py, &bytes);
        Ok((cls.into_any(), (py_bytes,)))
    }
}

/// Native interpretation of the Python `preconditioner` argument.
///
/// Mirrors `within`'s own `PrecondInput`: a prebuilt `Preconditioner`
/// takes the reuse path; everything else is an `Option<PreconditionerConfig>`
/// (None ⇒ upstream library default = additive Schwarz) that `Solver::new`
/// can build from directly.
enum PrecondInput {
    Prebuilt(within::Preconditioner),
    Config(Option<within::PreconditionerConfig>),
}

fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    (0..x.ncols())
        .map(|col| x.column(col).iter().copied().collect())
        .collect()
}

fn resolve_precond_input(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<PrecondInput> {
    let Some(preconditioner) = preconditioner else {
        return Ok(PrecondInput::Config(None));
    };

    if let Ok(s) = preconditioner.extract::<&str>() {
        return match s {
            "additive" => Ok(PrecondInput::Config(None)),
            "off" => Ok(PrecondInput::Config(Some(
                within::PreconditionerConfig::Off,
            ))),
            "diagonal" => Ok(PrecondInput::Config(Some(
                within::PreconditionerConfig::Diagonal,
            ))),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "preconditioner={other:?} is not supported by the 'within' \
                 LSMR backend; use 'additive' (default), 'off', 'diagonal', \
                 or a Preconditioner instance."
            ))),
        };
    }

    if let Ok(pre) = preconditioner.downcast::<PyPreconditioner>() {
        return Ok(PrecondInput::Prebuilt(pre.get().inner.clone()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "`preconditioner` must be 'additive', 'off', 'diagonal', or a Preconditioner instance.",
    ))
}

fn demean_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<u32>,
    weights: Option<&ArrayView1<f64>>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    preconditioner: PrecondInput,
) -> Result<(Array2<f64>, bool, Option<within::Preconditioner>), within::WithinError> {
    let n_obs = x.nrows();
    let n_rhs = x.ncols();

    let x_columns = extract_columns(x);
    let x_slices: Vec<&[f64]> = x_columns.iter().map(|col| col.as_slice()).collect();
    let weights_vec: Option<Vec<f64>> = weights.map(|w| w.iter().copied().collect());

    let options = within::LsmrOptions {
        tol,
        maxiter,
        local_size,
    };

    let solver = match preconditioner {
        PrecondInput::Prebuilt(pre) => {
            within::Solver::new(flist.view(), weights_vec, pre)?
        }
        PrecondInput::Config(cfg) => {
            within::Solver::new(flist.view(), weights_vec, cfg.as_ref())?
        }
    };

    let result = solver.solve_batch(&x_slices, &options)?;
    let preconditioner = solver.preconditioner().cloned();

    let all_converged = result.converged.iter().all(|&c| c);
    let out = Array2::from_shape_vec((n_obs, n_rhs).f(), result.demeaned)
        .expect("within returns one demeaned column per RHS");
    Ok((out, all_converged, preconditioner))
}

#[pyfunction]
#[pyo3(signature = (
    x,
    flist,
    weights=None,
    tol=1e-8,
    maxiter=1_000,
    local_size=None,
    preconditioner=None
))]
pub fn _demean_within_rs(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    flist: PyReadonlyArray2<u32>,
    weights: Option<PyReadonlyArray1<f64>>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Py<PyArray2<f64>>, bool, Option<Py<PyPreconditioner>>)> {
    let preconditioner = resolve_precond_input(preconditioner)?;

    let x_arr = x.as_array();
    let flist_arr = flist.as_array();
    let weights_arr = weights.as_ref().map(|w| w.as_array());

    let (out, success, built) = py
        .detach(|| {
            demean_within_impl(
                &x_arr,
                &flist_arr,
                weights_arr.as_ref(),
                tol,
                maxiter,
                local_size,
                preconditioner,
            )
        })
        .map_err(|e| match e {
            within::WithinError::Build(b) => {
                pyo3::exceptions::PyValueError::new_err(b.to_string())
            }
            within::WithinError::Solve(s) => {
                pyo3::exceptions::PyRuntimeError::new_err(s.to_string())
            }
            other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
        })?;

    let pyarray = PyArray2::from_owned_array(py, out);
    let py_preconditioner = match built {
        Some(inner) => Some(Py::new(py, PyPreconditioner { inner })?),
        None => None,
    };
    Ok((pyarray.into(), success, py_preconditioner))
}

pub fn add_pyclass(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreconditioner>()?;
    Ok(())
}
