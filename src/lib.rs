use pyo3::prelude::*;

mod collinear;
mod crv1;
mod demean;
mod demean_within;
mod nested_fixed_effects;
mod detect_singletons;
mod nw;

#[pymodule]
fn _core_impl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(collinear::_find_collinear_variables_rs))?;
    m.add_wrapped(wrap_pyfunction!(crv1::_crv1_meat_loop_rs))?;
    m.add_wrapped(wrap_pyfunction!(demean::_demean_rs))?;
    m.add_wrapped(wrap_pyfunction!(demean_within::_demean_within_rs))?;
    m.add_wrapped(wrap_pyfunction!(
        nested_fixed_effects::_count_fixef_fully_nested_all_rs
    ))?;
    m.add_wrapped(wrap_pyfunction!(detect_singletons::_detect_singletons_rs))?;
    m.add_wrapped(wrap_pyfunction!(nw::_nw_meat_panel_rs))?;
    m.add_wrapped(wrap_pyfunction!(nw::_nw_meat_time_rs))?;
    m.add_wrapped(wrap_pyfunction!(nw::_dk_meat_panel_rs))?;
    Ok(())
}
