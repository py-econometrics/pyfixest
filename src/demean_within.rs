use ndarray::{Array2, ArrayView1, ArrayView2, ShapeBuilder};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

/// Opaque handle to a pre-built within Schwarz preconditioner.
///
/// Two instances compare equal when their serialized representations match
/// — i.e. they describe the same factorization. ``__hash__`` is consistent
/// with ``__eq__``.
#[pyclass(frozen, name = "WithinPreconditioner", module = "pyfixest.core._core_impl")]
#[derive(Clone)]
pub struct PyWithinPreconditioner {
    inner: within::Preconditioner,
}

fn serialize_preconditioner(inner: &within::Preconditioner) -> PyResult<Vec<u8>> {
    postcard::to_stdvec(inner).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to serialize preconditioner: {}",
            e
        ))
    })
}

#[pymethods]
impl PyWithinPreconditioner {
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

    fn __repr__(&self) -> String {
        format!(
            "WithinPreconditioner(nrows={}, ncols={})",
            self.inner.nrows(),
            self.inner.ncols()
        )
    }

    fn __eq__(&self, other: &Self) -> PyResult<bool> {
        // Identity shortcut covers the IWLS reuse hot path without
        // re-serializing on every comparison.
        if std::ptr::eq(self, other) {
            return Ok(true);
        }
        if self.inner.nrows() != other.inner.nrows()
            || self.inner.ncols() != other.inner.ncols()
        {
            return Ok(false);
        }
        let a = serialize_preconditioner(&self.inner)?;
        let b = serialize_preconditioner(&other.inner)?;
        Ok(a == b)
    }

    fn __hash__(&self) -> PyResult<isize> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let bytes = serialize_preconditioner(&self.inner)?;
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = serialize_preconditioner(&self.inner)?;
        let cls = py.get_type::<Self>();
        let py_bytes = PyBytes::new(py, &bytes);
        Ok((cls.into_any(), (py_bytes,)))
    }
}

enum PreconditionerArg {
    Default,
    Off,
    Prebuilt(within::Preconditioner),
}

fn extract_columns(x: &ArrayView2<f64>) -> Vec<Vec<f64>> {
    (0..x.ncols())
        .map(|col| x.column(col).iter().copied().collect())
        .collect()
}

fn extract_preconditioner(
    preconditioner: Option<&Bound<'_, PyAny>>,
) -> PyResult<PreconditionerArg> {
    let Some(preconditioner) = preconditioner else {
        return Ok(PreconditionerArg::Default);
    };

    if let Ok(s) = preconditioner.extract::<&str>() {
        return match s {
            "schwarz" => Ok(PreconditionerArg::Default),
            "none" => Ok(PreconditionerArg::Off),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "preconditioner={other:?} is not supported by the 'within' \
                 LSMR backend; use 'schwarz' (default), 'none', or a \
                 WithinPreconditioner instance."
            ))),
        };
    }

    if let Ok(pre) = preconditioner.downcast::<PyWithinPreconditioner>() {
        return Ok(PreconditionerArg::Prebuilt(pre.get().inner.clone()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "`preconditioner` must be 'schwarz', 'none', or a WithinPreconditioner instance.",
    ))
}

fn demean_within_impl(
    x: &ArrayView2<f64>,
    flist: &ArrayView2<u32>,
    weights: Option<&ArrayView1<f64>>,
    tol: f64,
    maxiter: usize,
    local_size: Option<usize>,
    preconditioner: PreconditionerArg,
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
        PreconditionerArg::Default => within::Solver::new(
            flist.view(),
            weights_vec,
            Option::<&within::PreconditionerConfig>::None,
        )?,
        PreconditionerArg::Off => within::Solver::new(
            flist.view(),
            weights_vec,
            within::PreconditionerConfig::Off,
        )?,
        PreconditionerArg::Prebuilt(pre) => {
            within::Solver::new(flist.view(), weights_vec, pre)?
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
) -> PyResult<(Py<PyArray2<f64>>, bool, Option<Py<PyWithinPreconditioner>>)> {
    let preconditioner = extract_preconditioner(preconditioner)?;

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
        Some(inner) => Some(Py::new(py, PyWithinPreconditioner { inner })?),
        None => None,
    };
    Ok((pyarray.into(), success, py_preconditioner))
}

pub fn add_pyclass(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWithinPreconditioner>()?;
    Ok(())
}
