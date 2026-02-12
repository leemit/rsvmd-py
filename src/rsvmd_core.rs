use std::collections::VecDeque;

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::scale_space;
use crate::sliding_dft::SlidingDft;
use crate::vmd_core::{VmdConfig, VmdSolver, VmdState};

/// Output from RSVMD processing.
#[derive(Debug)]
pub struct RsvmdOutput {
    /// Decomposed modes in time domain, shape K x window_len.
    pub modes: Vec<Vec<f64>>,
    /// Center frequencies, length K.
    pub center_freqs: Vec<f64>,
    /// Number of ADMM iterations used.
    pub iterations: usize,
    /// Whether ADMM converged.
    pub converged: bool,
}

/// RSVMD processor: sliding VMD with recursive DFT and warm-starting.
pub struct RsvmdProcessor {
    config: VmdConfig,
    solver: VmdSolver,
    sdft: Option<SlidingDft>,
    state: VmdState,
    window_buffer: VecDeque<f64>,
}

impl RsvmdProcessor {
    pub fn new(config: VmdConfig) -> Self {
        let solver = VmdSolver::new(config.clone());
        let state = VmdState::new(config.k, config.window_len);

        RsvmdProcessor {
            config,
            solver,
            sdft: None,
            state,
            window_buffer: VecDeque::new(),
        }
    }

    /// Cold start: accepts exactly window_len samples.
    /// Computes full FFT, runs scale-space peak picking, runs VMD to convergence.
    pub fn initialize(&mut self, window: &[f64]) -> Result<RsvmdOutput, String> {
        let n = self.config.window_len;
        if window.len() != n {
            return Err(format!(
                "Initial window must be exactly {} samples, got {}",
                n, window.len()
            ));
        }

        // Store window buffer
        self.window_buffer.clear();
        self.window_buffer.extend(window.iter());

        // Initialize SDFT
        self.sdft = Some(SlidingDft::new(
            window,
            self.config.damping,
            self.config.fft_reset_interval,
        ));

        let signal_spectrum: Vec<Complex64> = self.sdft.as_ref()
            .ok_or("SDFT not initialized")?.spectrum().to_vec();

        // Scale-space peak picking for initial center frequencies
        let power_spectrum: Vec<f64> = signal_spectrum.iter().map(|c| c.norm_sqr()).collect();
        let picker = scale_space::default_peak_picker();
        let init_freqs = picker.pick_peaks(&power_spectrum, self.config.k);

        // Initialize state
        self.state = VmdState::new(self.config.k, n);
        self.state.center_freqs = init_freqs;
        self.state.initialized = true;

        // Run VMD ADMM
        let result = self.solver.solve(&signal_spectrum, &mut self.state);

        // Convert modes to time domain
        let modes = self.modes_to_time_domain(&result.mode_spectra);

        Ok(RsvmdOutput {
            modes,
            center_freqs: result.center_freqs,
            iterations: result.iterations,
            converged: result.converged,
        })
    }

    /// Warm update: accepts exactly step_size samples.
    /// Uses sliding DFT, warm-starts ADMM from previous state.
    pub fn update(&mut self, new_samples: &[f64]) -> Result<RsvmdOutput, String> {
        if !self.state.initialized {
            // If not initialized and we receive window_len samples, do cold start
            if new_samples.len() == self.config.window_len {
                return self.initialize(new_samples);
            }
            return Err(format!(
                "Processor not initialized. First call must provide {} samples.",
                self.config.window_len
            ));
        }

        let s = new_samples.len();
        if s != self.config.step_size {
            return Err(format!(
                "Expected {} samples, got {}",
                self.config.step_size, s
            ));
        }

        // Get old samples that are leaving the window
        let old_samples: Vec<f64> = self.window_buffer.iter().take(s).copied().collect();

        // Update window buffer
        for _ in 0..s {
            self.window_buffer.pop_front();
        }
        for &sample in new_samples {
            self.window_buffer.push_back(sample);
        }

        // Sliding DFT update
        let sdft = self.sdft.as_mut().ok_or("SDFT not initialized")?;
        sdft.slide_block(new_samples, &old_samples)?;

        // Check if FFT reset is needed
        if sdft.needs_reset() {
            let buf: Vec<f64> = self.window_buffer.iter().copied().collect();
            let _ = sdft.reset_from_buffer(&buf);
        }

        let signal_spectrum: Vec<Complex64> = sdft.spectrum().to_vec();

        // Run warm-started ADMM
        let result = self.solver.solve(&signal_spectrum, &mut self.state);

        // Convert to time domain
        let modes = self.modes_to_time_domain(&result.mode_spectra);

        Ok(RsvmdOutput {
            modes,
            center_freqs: result.center_freqs,
            iterations: result.iterations,
            converged: result.converged,
        })
    }

    /// Get current center frequencies.
    pub fn center_freqs(&self) -> &[f64] {
        &self.state.center_freqs
    }

    /// Check if initialized.
    pub fn initialized(&self) -> bool {
        self.state.initialized
    }

    /// Force FFT reset.
    pub fn reset_fft(&mut self) {
        if let Some(ref mut sdft) = self.sdft {
            let buf: Vec<f64> = self.window_buffer.iter().copied().collect();
            let _ = sdft.reset_from_buffer(&buf);
        }
    }

    /// Convert mode spectra to time domain via inverse FFT.
    fn modes_to_time_domain(&self, mode_spectra: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
        let n = self.config.window_len;
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);

        mode_spectra
            .iter()
            .map(|spectrum| {
                let mut buffer = spectrum.clone();
                ifft.process(&mut buffer);
                // Normalize by N (rustfft doesn't normalize)
                buffer.iter().map(|c| c.re / n as f64).collect()
            })
            .collect()
    }

    /// Access internal state (for PO-RSVMD).
    pub fn state(&self) -> &VmdState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut VmdState {
        &mut self.state
    }

    pub fn solver(&self) -> &VmdSolver {
        &self.solver
    }

    pub fn sdft(&self) -> Option<&SlidingDft> {
        self.sdft.as_ref()
    }

    pub fn sdft_mut(&mut self) -> Option<&mut SlidingDft> {
        self.sdft.as_mut()
    }

    pub fn config(&self) -> &VmdConfig {
        &self.config
    }

    pub fn window_buffer(&self) -> &VecDeque<f64> {
        &self.window_buffer
    }

    pub fn window_buffer_mut(&mut self) -> &mut VecDeque<f64> {
        &mut self.window_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_signal(n: usize) -> Vec<f64> {
        let dt = 1.0 / n as f64;
        (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.7 * (2.0 * PI * 20.0 * t).sin()
                    + 0.5 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_cold_start() {
        let n = 512;
        let signal = make_signal(n);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 3);
        assert_eq!(output.modes[0].len(), n);
        assert_eq!(output.center_freqs.len(), 3);
        assert!(proc.initialized());
    }

    #[test]
    fn test_warm_update() {
        let n = 256;
        let total_samples = n + 10;
        let signal = make_signal(total_samples);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            step_size: 1,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);

        // Cold start
        let cold = proc.initialize(&signal[..n]).unwrap();
        let cold_iters = cold.iterations;

        // Warm updates — verify they produce valid output
        for i in 0..10 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            assert_eq!(output.modes.len(), 3);
            assert_eq!(output.modes[0].len(), n);
            assert_eq!(output.center_freqs.len(), 3);
        }

        // Cold start with tau > 0 should take multiple iterations
        assert!(
            cold_iters > 1,
            "Cold start should take multiple iterations, got {}",
            cold_iters
        );
    }

    #[test]
    fn test_auto_initialize_on_first_update() {
        let n = 256;
        let signal = make_signal(n);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);

        // Pass window_len samples to update — should auto-initialize
        let output = proc.update(&signal).unwrap();
        assert!(proc.initialized());
        assert_eq!(output.modes.len(), 3);
    }

    #[test]
    fn test_initialize_wrong_length_returns_err() {
        let config = VmdConfig {
            window_len: 256,
            k: 3,
            ..Default::default()
        };
        let mut proc = RsvmdProcessor::new(config);
        let result = proc.initialize(&[0.0; 100]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exactly 256 samples"));
    }

    #[test]
    fn test_update_before_init_returns_err() {
        let config = VmdConfig {
            window_len: 256,
            k: 3,
            step_size: 1,
            ..Default::default()
        };
        let mut proc = RsvmdProcessor::new(config);
        let result = proc.update(&[0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not initialized"));
    }

    #[test]
    fn test_update_wrong_step_size_returns_err() {
        let n = 128;
        let signal = make_signal(n);
        let config = VmdConfig {
            window_len: n,
            k: 2,
            step_size: 1,
            ..Default::default()
        };
        let mut proc = RsvmdProcessor::new(config);
        proc.initialize(&signal).unwrap();

        let result = proc.update(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 1 samples"));
    }

    #[test]
    fn test_time_domain_reconstruction_quality() {
        let n = 512;
        let signal = make_signal(n);

        // Use tau > 0 to enforce reconstruction constraint via dual variable
        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        // Sum of modes in time domain should capture most signal energy
        let mut reconstructed = vec![0.0; n];
        for mode in &output.modes {
            for i in 0..n {
                reconstructed[i] += mode[i];
            }
        }

        let mut err_sq = 0.0;
        let mut sig_sq = 0.0;
        for i in 0..n {
            err_sq += (reconstructed[i] - signal[i]).powi(2);
            sig_sq += signal[i].powi(2);
        }
        let relative_error = (err_sq / sig_sq).sqrt();

        assert!(
            relative_error < 1.0,
            "Time-domain reconstruction error too high: {:.4}",
            relative_error
        );
    }

    #[test]
    fn test_streaming_reconstruction_stability() {
        let n = 256;
        let total = n + 20;
        let signal = make_signal(total);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            step_size: 1,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        // Track reconstruction errors across frames
        let mut errors = Vec::new();
        for i in 0..20 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            let window = &signal[i + 1..i + 1 + n];

            let mut reconstructed = vec![0.0; n];
            for mode in &output.modes {
                for j in 0..n {
                    reconstructed[j] += mode[j];
                }
            }

            let mut err_sq = 0.0;
            let mut sig_sq = 0.0;
            for j in 0..n {
                err_sq += (reconstructed[j] - window[j]).powi(2);
                sig_sq += window[j].powi(2);
            }
            errors.push((err_sq / sig_sq).sqrt());
        }

        // Errors should not blow up over time
        for (i, &e) in errors.iter().enumerate() {
            assert!(
                e < 1.0,
                "Reconstruction error at frame {} is too high: {:.4}",
                i, e
            );
        }

        // Error should not grow monotonically (stability check)
        let last = errors[errors.len() - 1];
        let first = errors[0];
        assert!(
            last < first * 3.0,
            "Error grew too much: first={:.4}, last={:.4}",
            first, last
        );
    }

    #[test]
    fn test_single_mode_k1() {
        let n = 256;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 20.0 * i as f64 * dt).sin())
            .collect();

        let config = VmdConfig {
            alpha: 2000.0,
            k: 1,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 1);
        assert_eq!(output.modes[0].len(), n);

        // Single mode should capture most of the signal energy
        let mut err_sq = 0.0;
        let mut sig_sq = 0.0;
        for i in 0..n {
            err_sq += (output.modes[0][i] - signal[i]).powi(2);
            sig_sq += signal[i].powi(2);
        }
        let relative_error = (err_sq / sig_sq).sqrt();
        assert!(
            relative_error < 1.0,
            "K=1 reconstruction error too high: {:.4}",
            relative_error
        );

        // No NaN values
        for &v in &output.modes[0] {
            assert!(!v.is_nan(), "K=1 mode has NaN");
        }
    }

    #[test]
    fn test_large_k_more_modes_than_content() {
        let n = 256;
        let dt = 1.0 / n as f64;
        // Only 2 sinusoids, but K=5
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + 0.5 * (2.0 * PI * 80.0 * t).sin()
            })
            .collect();

        let config = VmdConfig {
            alpha: 2000.0,
            k: 5,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        // Should not crash, produce 5 modes
        assert_eq!(output.modes.len(), 5);
        assert_eq!(output.center_freqs.len(), 5);

        // No NaN in output
        for freq in &output.center_freqs {
            assert!(!freq.is_nan(), "Center freq is NaN");
        }
        for mode in &output.modes {
            for &v in mode {
                assert!(!v.is_nan(), "Mode value is NaN");
            }
        }
    }

    #[test]
    fn test_zero_signal() {
        let n = 128;
        let signal = vec![0.0; n];

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 2);
        // No NaN
        for mode in &output.modes {
            for &v in mode {
                assert!(!v.is_nan(), "Mode value is NaN on zero signal");
            }
        }
        for &f in &output.center_freqs {
            assert!(!f.is_nan(), "Center freq is NaN on zero signal");
        }
    }

    #[test]
    fn test_constant_signal() {
        let n = 128;
        let signal = vec![3.14; n];

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 2);
        for mode in &output.modes {
            for &v in mode {
                assert!(!v.is_nan(), "Mode value is NaN on constant signal");
            }
        }
    }

    #[test]
    fn test_step_size_greater_than_one() {
        let n = 256;
        let step = 5;
        let total = n + step * 5;
        let signal = make_signal(total);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            step_size: step,
            max_iter: 500,
            ..Default::default()
        };

        let mut proc = RsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        for i in 0..5 {
            let start = n + i * step;
            let output = proc.update(&signal[start..start + step]).unwrap();
            assert_eq!(output.modes.len(), 3);
            assert_eq!(output.modes[0].len(), n);
        }
    }

    #[test]
    fn test_fft_reset_interval() {
        let n = 128;
        let total = n + 20;
        let signal = make_signal(total);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.0,
            tol: 1e-7,
            window_len: n,
            step_size: 1,
            max_iter: 500,
            damping: 0.99999,
            fft_reset_interval: 5,
        };

        let mut proc = RsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        // Run through frames that cross the reset interval
        for i in 0..20 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            assert_eq!(output.modes.len(), 2);
            for mode in &output.modes {
                for &v in mode {
                    assert!(!v.is_nan(), "NaN after FFT reset at frame {}", i);
                }
            }
        }
    }
}
