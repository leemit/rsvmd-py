use std::collections::VecDeque;

use num_complex::Complex64;
use rustfft::FftPlanner;

use crate::scale_space;
use crate::sliding_dft::SlidingDft;
use crate::vmd_core::{VmdConfig, VmdSolver, VmdState};

/// Output from RSVMD processing.
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
    pub fn initialize(&mut self, window: &[f64]) -> RsvmdOutput {
        let n = self.config.window_len;
        assert_eq!(
            window.len(),
            n,
            "Initial window must be exactly {} samples",
            n
        );

        // Store window buffer
        self.window_buffer.clear();
        self.window_buffer.extend(window.iter());

        // Initialize SDFT
        self.sdft = Some(SlidingDft::new(
            window,
            self.config.damping,
            self.config.fft_reset_interval,
        ));

        let signal_spectrum: Vec<Complex64> = self.sdft.as_ref().unwrap().spectrum().to_vec();

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

        RsvmdOutput {
            modes,
            center_freqs: result.center_freqs,
            iterations: result.iterations,
            converged: result.converged,
        }
    }

    /// Warm update: accepts exactly step_size samples.
    /// Uses sliding DFT, warm-starts ADMM from previous state.
    pub fn update(&mut self, new_samples: &[f64]) -> RsvmdOutput {
        if !self.state.initialized {
            // If not initialized and we receive window_len samples, do cold start
            if new_samples.len() == self.config.window_len {
                return self.initialize(new_samples);
            }
            panic!(
                "Processor not initialized. First call must provide {} samples.",
                self.config.window_len
            );
        }

        let s = new_samples.len();
        assert_eq!(
            s, self.config.step_size,
            "Expected {} samples, got {}",
            self.config.step_size, s
        );

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
        let sdft = self.sdft.as_mut().unwrap();
        sdft.slide_block(new_samples, &old_samples);

        // Check if FFT reset is needed
        if sdft.needs_reset() {
            let buf: Vec<f64> = self.window_buffer.iter().copied().collect();
            sdft.reset_from_buffer(&buf);
        }

        let signal_spectrum: Vec<Complex64> = sdft.spectrum().to_vec();

        // Run warm-started ADMM
        let result = self.solver.solve(&signal_spectrum, &mut self.state);

        // Convert to time domain
        let modes = self.modes_to_time_domain(&result.mode_spectra);

        RsvmdOutput {
            modes,
            center_freqs: result.center_freqs,
            iterations: result.iterations,
            converged: result.converged,
        }
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
            sdft.reset_from_buffer(&buf);
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
        let output = proc.initialize(&signal);

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
        let cold = proc.initialize(&signal[..n]);
        let cold_iters = cold.iterations;

        // Warm updates — verify they produce valid output
        for i in 0..10 {
            let output = proc.update(&signal[n + i..n + i + 1]);
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
        let output = proc.update(&signal);
        assert!(proc.initialized());
        assert_eq!(output.modes.len(), 3);
    }
}
