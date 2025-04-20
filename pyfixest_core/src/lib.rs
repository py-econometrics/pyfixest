use pyo3::prelude::*;

// bring in your sub‑modules
mod collinear;
mod crv1;
mod demean;

#[pymodule]
fn pyfixest_core(py: Python, m: &PyModule) -> PyResult<()> {
    // wrap the functions you want exposed
    m.add_function(wrap_pyfunction!(collinear::find_collinear_variables, m)?)?;
    m.add_function(wrap_pyfunction!(crv1::crv1_meat_loop, m)?)?;
    m.add_function(wrap_pyfunction!(demean::demean,     m)?)?;
    Ok(())
}
