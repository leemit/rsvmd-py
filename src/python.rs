// This work is dedicated to the public domain under the CC0 1.0 Universal license.
// To the extent possible under law, the author has waived all copyright
// and related or neighboring rights to this work.
// https://creativecommons.org/publicdomain/zero/1.0/

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::po_rsvmd::{PoRsvmdConfig, PoRsvmdProcessor as RustPoRsvmdProcessor};
use crate::rsvmd_core::RsvmdProcessor as RustRsvmdProcessor;
use crate::vmd_core::VmdConfig;

#[pyclass(name = "RSVMDProcessor")]
pub struct PyRSVMDProcessor {
    inner: RustRsvmdProcessor,
    last_iterations: usize,
    last_converged: bool,
}

#[pymethods]
impl PyRSVMDProcessor {
    #[new]
    #[pyo3(signature = (alpha=2000.0, k=3, tau=0.0, tol=1e-7, window_len=7200, step_size=1, max_iter=500, damping=0.99999, fft_reset_interval=0))]
    fn new(
        alpha: f64,
        k: usize,
        tau: f64,
        tol: f64,
        window_len: usize,
        step_size: usize,
        max_iter: usize,
        damping: f64,
        fft_reset_interval: usize,
    ) -> Self {
        let config = VmdConfig {
            alpha,
            k,
            tau,
            tol,
            window_len,
            step_size,
            max_iter,
            damping,
            fft_reset_interval,
        };
        PyRSVMDProcessor {
            inner: RustRsvmdProcessor::new(config),
            last_iterations: 0,
            last_converged: false,
        }
    }

    /// Process samples. First call = cold start (len must == window_len).
    /// Subsequent calls = warm update (len must == step_size).
    /// Returns (modes: ndarray[K, window_len], center_freqs: ndarray[K])
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        samples: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        let slice = samples.as_slice()?;

        let output = if !self.inner.initialized() {
            if slice.len() == self.inner.config().window_len {
                self.inner.initialize(slice)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "First call must provide exactly {} samples (window_len), got {}",
                    self.inner.config().window_len,
                    slice.len()
                )));
            }
        } else {
            if slice.len() != self.inner.config().step_size {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Expected {} samples (step_size), got {}",
                    self.inner.config().step_size,
                    slice.len()
                )));
            }
            self.inner.update(slice)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
        };

        self.last_iterations = output.iterations;
        self.last_converged = output.converged;

        let k = output.modes.len();
        let n = if k > 0 { output.modes[0].len() } else { 0 };

        // Create 2D numpy array for modes (K x N)
        let modes_flat: Vec<f64> = output.modes.into_iter().flatten().collect();
        let modes_array = PyArray1::from_vec(py, modes_flat)
            .reshape([k, n])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let freqs_array = PyArray1::from_vec(py, output.center_freqs);

        Ok((modes_array, freqs_array))
    }

    /// Get current center frequencies.
    fn center_freqs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.center_freqs())
    }

    /// Force FFT reset.
    fn reset_fft(&mut self) {
        self.inner.reset_fft();
    }

    /// Check if processor has been initialized (cold start done).
    #[getter]
    fn initialized(&self) -> bool {
        self.inner.initialized()
    }

    /// Number of ADMM iterations used in the last update call.
    #[getter]
    fn last_iterations(&self) -> usize {
        self.last_iterations
    }

    /// Whether the last update call converged.
    #[getter]
    fn last_converged(&self) -> bool {
        self.last_converged
    }
}

#[pyclass(name = "PORSVMDProcessor")]
pub struct PyPORSVMDProcessor {
    inner: RustPoRsvmdProcessor,
    window_len: usize,
    step_size: usize,
    last_iterations: usize,
    last_converged: bool,
}

#[pymethods]
impl PyPORSVMDProcessor {
    #[new]
    #[pyo3(signature = (alpha=2000.0, k=3, tau=0.0, tol=1e-7, window_len=7200, step_size=1, max_iter=500, damping=0.99999, fft_reset_interval=0, gamma_default=0.5, gamma_tiers=None))]
    fn new(
        alpha: f64,
        k: usize,
        tau: f64,
        tol: f64,
        window_len: usize,
        step_size: usize,
        max_iter: usize,
        damping: f64,
        fft_reset_interval: usize,
        gamma_default: f64,
        gamma_tiers: Option<Vec<(f64, f64)>>,
    ) -> Self {
        let base_config = VmdConfig {
            alpha,
            k,
            tau,
            tol,
            window_len,
            step_size,
            max_iter,
            damping,
            fft_reset_interval,
        };
        let mut po_config = PoRsvmdConfig::new(base_config, gamma_default);
        if let Some(tiers) = gamma_tiers {
            po_config.gamma_tiers = tiers;
        }

        PyPORSVMDProcessor {
            inner: RustPoRsvmdProcessor::new(po_config),
            window_len,
            step_size,
            last_iterations: 0,
            last_converged: false,
        }
    }

    /// Process samples. Same interface as RSVMDProcessor.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        samples: PyReadonlyArray1<f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        let slice = samples.as_slice()?;

        let output = if !self.inner.initialized() {
            if slice.len() != self.window_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "First call must provide exactly {} samples (window_len), got {}",
                    self.window_len,
                    slice.len()
                )));
            }
            self.inner.initialize(slice)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
        } else {
            if slice.len() != self.step_size {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Expected {} samples (step_size), got {}",
                    self.step_size,
                    slice.len()
                )));
            }
            self.inner.update(slice)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?
        };

        self.last_iterations = output.iterations;
        self.last_converged = output.converged;

        let k = output.modes.len();
        let n = if k > 0 { output.modes[0].len() } else { 0 };

        let modes_flat: Vec<f64> = output.modes.into_iter().flatten().collect();
        let modes_array = PyArray1::from_vec(py, modes_flat)
            .reshape([k, n])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let freqs_array = PyArray1::from_vec(py, output.center_freqs);

        Ok((modes_array, freqs_array))
    }

    /// Get current center frequencies.
    fn center_freqs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, self.inner.center_freqs())
    }

    /// Force FFT reset.
    fn reset_fft(&mut self) {
        self.inner.reset_fft();
    }

    /// Get the current adaptive gamma value.
    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Check if processor has been initialized.
    #[getter]
    fn initialized(&self) -> bool {
        self.inner.initialized()
    }

    /// Number of ADMM iterations used in the last update call.
    #[getter]
    fn last_iterations(&self) -> usize {
        self.last_iterations
    }

    /// Whether the last update call converged.
    #[getter]
    fn last_converged(&self) -> bool {
        self.last_converged
    }
}

pub fn register_module(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyRSVMDProcessor>()?;
    m.add_class::<PyPORSVMDProcessor>()?;
    Ok(())
}
