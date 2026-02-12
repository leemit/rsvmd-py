use pyo3::prelude::*;

pub mod complex_utils;
pub mod po_rsvmd;
pub mod python;
pub mod rsvmd_core;
pub mod scale_space;
pub mod sliding_dft;
pub mod vmd_core;

#[pymodule(name = "_rsvmd")]
fn rsvmd_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
