use pyo3::prelude::*;

mod collinear;
mod crv1;
mod demean;
mod nested_fixed_effects;

#[pymodule]
fn _core_impl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collinear::_find_collinear_variables_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crv1::_crv1_meat_loop_rs, m)?)?;
    m.add_function(wrap_pyfunction!(demean::_demean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(
        nested_fixed_effects::_count_fixef_fully_nested_all_rs,
        m
    )?)?;
    Ok(())
}

pub use demean::demean_impl;
