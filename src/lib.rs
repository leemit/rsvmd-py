// This work is dedicated to the public domain under the CC0 1.0 Universal license.
// To the extent possible under law, the author has waived all copyright
// and related or neighboring rights to this work.
// https://creativecommons.org/publicdomain/zero/1.0/

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
