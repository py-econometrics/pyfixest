use pyo3::prelude::*;

mod collinear;
mod crv1;
mod demean;
mod nested_fixed_effects;

#[pymodule]
fn pyfixest_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collinear::find_collinear_variables_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crv1::crv1_meat_loop_rs, m)?)?;
    m.add_function(wrap_pyfunction!(demean::demean_rs,     m)?)?;
    m.add_function(wrap_pyfunction!(nested_fixed_effects::count_fixef_fully_nested_all_rs,     m)?)?;

    Ok(())
}
