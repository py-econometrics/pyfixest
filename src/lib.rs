use pyo3::prelude::*;

mod collinear;
mod crv1;
mod demean;
mod nested_fixed_effects;

#[pymodule]
fn _core_impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(collinear::_find_collinear_variables_rs))?;
    m.add_wrapped(wrap_pyfunction!(crv1::_crv1_meat_loop_rs))?;
    m.add_wrapped(wrap_pyfunction!(demean::_demean_rs))?;
    m.add_wrapped(wrap_pyfunction!(
        nested_fixed_effects::_count_fixef_fully_nested_all_rs
    ))?;
    Ok(())
}
