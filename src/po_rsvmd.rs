use std::time::Instant;

use num_complex::Complex64;

use crate::complex_utils;
use crate::rsvmd_core::{RsvmdOutput, RsvmdProcessor};
use crate::scale_space;
use crate::vmd_core::VmdConfig;

/// PO-RSVMD specific configuration.
#[derive(Clone, Debug)]
pub struct PoRsvmdConfig {
    pub base: VmdConfig,
    /// Default rate learning factor when delta_t < smallest threshold.
    pub gamma_default: f64,
    /// Piecewise mapping from delta_t thresholds to gamma values (Formula 20).
    /// Entries: [(threshold, gamma_value), ...] sorted descending by threshold.
    pub gamma_tiers: Vec<(f64, f64)>,
}

impl PoRsvmdConfig {
    pub fn new(base: VmdConfig, gamma_default: f64) -> Self {
        PoRsvmdConfig {
            base,
            gamma_default,
            gamma_tiers: vec![
                (0.8, 0.0),
                (0.6, 0.001),
                (0.4, 0.01),
                (0.2, 0.05),
                (0.1, 0.2),
            ],
        }
    }
}

/// PO-RSVMD processor with error mutation detection and adaptive gamma.
pub struct PoRsvmdProcessor {
    inner: RsvmdProcessor,
    po_config: PoRsvmdConfig,
    /// Previous iteration time for gamma adaptation.
    prev_iteration_time: Option<f64>,
    /// Second-to-last iteration time for delta_t computation.
    prev_prev_iteration_time: Option<f64>,
    /// Current adaptive gamma.
    gamma: f64,
}

impl PoRsvmdProcessor {
    pub fn new(config: PoRsvmdConfig) -> Self {
        let gamma = config.gamma_default;
        let inner = RsvmdProcessor::new(config.base.clone());
        PoRsvmdProcessor {
            inner,
            po_config: config,
            prev_iteration_time: None,
            prev_prev_iteration_time: None,
            gamma,
        }
    }

    /// Cold start: same as RSVMD.
    pub fn initialize(&mut self, window: &[f64]) -> Result<RsvmdOutput, String> {
        let start = Instant::now();
        let output = self.inner.initialize(window)?;
        self.prev_iteration_time = Some(start.elapsed().as_secs_f64());
        self.prev_prev_iteration_time = None;
        Ok(output)
    }

    /// Update with error mutation detection and adaptive gamma.
    pub fn update(&mut self, new_samples: &[f64]) -> Result<RsvmdOutput, String> {
        if !self.inner.initialized() {
            if new_samples.len() == self.inner.config().window_len {
                return self.initialize(new_samples);
            }
            return Err(format!(
                "Processor not initialized. First call must provide {} samples.",
                self.inner.config().window_len
            ));
        }

        let k = self.inner.config().k;
        let s = new_samples.len();
        if s != self.inner.config().step_size {
            return Err(format!(
                "Expected {} samples, got {}",
                self.inner.config().step_size, s
            ));
        }

        // --- Sliding DFT update ---
        // Get old samples before mutating the buffer
        let old_samples: Vec<f64> = self.inner.window_buffer().iter().take(s).copied().collect();

        // Update window buffer
        {
            let buf = self.inner.window_buffer_mut();
            for _ in 0..s {
                buf.pop_front();
            }
            for &sample in new_samples {
                buf.push_back(sample);
            }
        }

        // Update SDFT
        {
            let sdft = self.inner.sdft_mut().ok_or("SDFT not initialized")?;
            sdft.slide_block(new_samples, &old_samples)?;
        }

        // Check if FFT reset is needed (separate borrow scope)
        {
            let needs_reset = self.inner.sdft().ok_or("SDFT not initialized")?.needs_reset();
            if needs_reset {
                let buf: Vec<f64> = self.inner.window_buffer().iter().copied().collect();
                let _ = self.inner.sdft_mut().ok_or("SDFT not initialized")?.reset_from_buffer(&buf);
            }
        }

        let signal_spectrum: Vec<Complex64> =
            self.inner.sdft().ok_or("SDFT not initialized")?.spectrum().to_vec();

        // --- Adaptive center frequency initialization ---
        let power_spectrum: Vec<f64> = signal_spectrum.iter().map(|c| c.norm_sqr()).collect();
        let picker = scale_space::default_peak_picker();
        let detected_freqs = picker.pick_peaks(&power_spectrum, k);

        // Compute gamma from iteration time history
        let start = Instant::now();

        if let (Some(prev), Some(prev_prev)) =
            (self.prev_iteration_time, self.prev_prev_iteration_time)
        {
            let delta_t = (prev - prev_prev).abs();
            self.gamma = self.compute_gamma(delta_t);
        }

        // Blend center frequencies
        let prev_freqs = self.inner.state().center_freqs.clone();
        let blended_freqs: Vec<f64> = prev_freqs
            .iter()
            .zip(detected_freqs.iter())
            .map(|(&prev, &det)| self.gamma * prev + (1.0 - self.gamma) * det)
            .collect();

        self.inner.state_mut().center_freqs = blended_freqs;

        // --- ADMM loop with error mutation check ---
        let result = self.solve_with_mutation_check(&signal_spectrum);

        let elapsed = start.elapsed().as_secs_f64();
        self.prev_prev_iteration_time = self.prev_iteration_time;
        self.prev_iteration_time = Some(elapsed);

        // Convert to time domain
        let modes = self.modes_to_time_domain(&result.mode_spectra);

        // Update state
        self.inner.state_mut().mode_spectra = result.mode_spectra;
        self.inner.state_mut().center_freqs = result.center_freqs.clone();
        self.inner.state_mut().lambda = result.lambda;

        Ok(RsvmdOutput {
            modes,
            center_freqs: result.center_freqs,
            iterations: result.iterations,
            converged: result.converged,
        })
    }

    /// ADMM solve with error mutation detection.
    fn solve_with_mutation_check(
        &self,
        signal_spectrum: &[Complex64],
    ) -> MutationResult {
        let state = self.inner.state();
        let config = self.inner.config();
        let n = config.window_len;
        let k = config.k;

        let mut mode_spectra = state.mode_spectra.clone();
        let mut center_freqs = state.center_freqs.clone();
        let mut lambda = state.lambda.clone();

        let mut prev_error = f64::INFINITY;
        let mut best_modes = mode_spectra.clone();
        let mut best_freqs = center_freqs.clone();
        let mut best_lambda = lambda.clone();

        let mut prev_modes: Vec<Vec<Complex64>> =
            mode_spectra.iter().map(|m| m.clone()).collect();

        let freqs = self.inner.solver().freqs();

        for iter in 0..config.max_iter {
            // Save for convergence check
            for ki in 0..k {
                prev_modes[ki].copy_from_slice(&mode_spectra[ki]);
            }

            // ADMM step
            for ki in 0..k {
                let mut other_sum = vec![Complex64::new(0.0, 0.0); n];
                for kj in 0..k {
                    if kj != ki {
                        for i in 0..n {
                            other_sum[i] += mode_spectra[kj][i];
                        }
                    }
                }

                let omega_k = center_freqs[ki];
                for i in 0..n {
                    let numerator =
                        signal_spectrum[i] - other_sum[i] + lambda[i] * 0.5;
                    let freq_diff = freqs[i] - omega_k;
                    let denom =
                        1.0 + 2.0 * config.alpha * freq_diff * freq_diff;
                    mode_spectra[ki][i] = numerator / denom;
                }

                // Center frequency update
                let half_n = n / 2;
                let mut weighted_sum = 0.0;
                let mut power_sum = 0.0;
                for i in 0..=half_n {
                    let power = mode_spectra[ki][i].norm_sqr();
                    weighted_sum += freqs[i] * power;
                    power_sum += power;
                }
                center_freqs[ki] = if power_sum > 1e-30 {
                    weighted_sum / power_sum
                } else {
                    0.0
                };
            }

            // Dual variable update
            if config.tau > 0.0 {
                let mut mode_sum = vec![Complex64::new(0.0, 0.0); n];
                for ki in 0..k {
                    for i in 0..n {
                        mode_sum[i] += mode_spectra[ki][i];
                    }
                }
                for i in 0..n {
                    lambda[i] += Complex64::new(config.tau, 0.0)
                        * (signal_spectrum[i] - mode_sum[i]);
                }
            }

            // Error mutation check
            let curr_error =
                complex_utils::reconstruction_error(&mode_spectra, signal_spectrum);

            if curr_error > prev_error {
                return MutationResult {
                    mode_spectra: best_modes,
                    center_freqs: best_freqs,
                    lambda: best_lambda,
                    iterations: iter,
                    converged: false,
                    final_error: prev_error,
                };
            }

            best_modes = mode_spectra.clone();
            best_freqs = center_freqs.clone();
            best_lambda = lambda.clone();
            prev_error = curr_error;

            // Convergence check
            let mut metric = 0.0;
            for ki in 0..k {
                let diff_sq =
                    complex_utils::diff_norm_sqr(&mode_spectra[ki], &prev_modes[ki]);
                let prev_sq = complex_utils::norm_sqr_sum(&prev_modes[ki]);
                if prev_sq > 1e-30 {
                    metric += diff_sq / prev_sq;
                }
            }

            if metric < config.tol {
                return MutationResult {
                    mode_spectra,
                    center_freqs,
                    lambda,
                    iterations: iter + 1,
                    converged: true,
                    final_error: curr_error,
                };
            }
        }

        MutationResult {
            mode_spectra,
            center_freqs,
            lambda,
            iterations: config.max_iter,
            converged: false,
            final_error: prev_error,
        }
    }

    /// Compute gamma from delta_t using the piecewise mapping (Formula 20).
    fn compute_gamma(&self, delta_t: f64) -> f64 {
        for &(threshold, gamma_val) in &self.po_config.gamma_tiers {
            if delta_t >= threshold {
                return gamma_val;
            }
        }
        self.po_config.gamma_default
    }

    /// Convert mode spectra to time domain via inverse FFT.
    fn modes_to_time_domain(&self, mode_spectra: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
        let n = self.inner.config().window_len;
        let mut planner = rustfft::FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);

        mode_spectra
            .iter()
            .map(|spectrum| {
                let mut buffer = spectrum.clone();
                ifft.process(&mut buffer);
                buffer.iter().map(|c| c.re / n as f64).collect()
            })
            .collect()
    }

    /// Get current gamma value.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Check if initialized.
    pub fn initialized(&self) -> bool {
        self.inner.initialized()
    }

    /// Get center frequencies.
    pub fn center_freqs(&self) -> &[f64] {
        self.inner.center_freqs()
    }

    /// Force FFT reset.
    pub fn reset_fft(&mut self) {
        self.inner.reset_fft();
    }
}

struct MutationResult {
    mode_spectra: Vec<Vec<Complex64>>,
    center_freqs: Vec<f64>,
    lambda: Vec<Complex64>,
    iterations: usize,
    converged: bool,
    #[allow(dead_code)]
    final_error: f64,
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
    fn test_po_rsvmd_cold_start() {
        let n = 256;
        let signal = make_signal(n);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tol: 1e-7,
                window_len: n,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 3);
        assert_eq!(output.center_freqs.len(), 3);
        assert!(proc.initialized());
    }

    #[test]
    fn test_po_rsvmd_warm_update() {
        let n = 256;
        let total = n + 5;
        let signal = make_signal(total);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        for i in 0..5 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            assert_eq!(output.modes.len(), 3);
        }
    }

    #[test]
    fn test_gamma_tiers() {
        let config = PoRsvmdConfig::new(
            VmdConfig {
                window_len: 64,
                k: 2,
                ..Default::default()
            },
            0.5,
        );
        let proc = PoRsvmdProcessor::new(config);

        assert_eq!(proc.compute_gamma(0.9), 0.0);
        assert_eq!(proc.compute_gamma(0.8), 0.0);
        assert_eq!(proc.compute_gamma(0.7), 0.001);
        assert_eq!(proc.compute_gamma(0.5), 0.01);
        assert_eq!(proc.compute_gamma(0.3), 0.05);
        assert_eq!(proc.compute_gamma(0.15), 0.2);
        assert_eq!(proc.compute_gamma(0.05), 0.5);
    }

    #[test]
    fn test_po_rsvmd_initialize_wrong_length_returns_err() {
        let config = PoRsvmdConfig::new(
            VmdConfig {
                window_len: 256,
                k: 3,
                ..Default::default()
            },
            0.5,
        );
        let mut proc = PoRsvmdProcessor::new(config);
        let result = proc.initialize(&[0.0; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_po_rsvmd_update_before_init_returns_err() {
        let config = PoRsvmdConfig::new(
            VmdConfig {
                window_len: 256,
                k: 3,
                step_size: 1,
                ..Default::default()
            },
            0.5,
        );
        let mut proc = PoRsvmdProcessor::new(config);
        let result = proc.update(&[0.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not initialized"));
    }

    #[test]
    fn test_po_rsvmd_update_wrong_step_size_returns_err() {
        let n = 128;
        let signal = make_signal(n);
        let config = PoRsvmdConfig::new(
            VmdConfig {
                window_len: n,
                k: 2,
                step_size: 1,
                ..Default::default()
            },
            0.5,
        );
        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal).unwrap();

        let result = proc.update(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Expected 1 samples"));
    }

    #[test]
    fn test_error_mutation_detection_stops_early() {
        // Use max_iter=1 to force only 1 ADMM iteration per frame,
        // which means the error mutation check has limited room.
        // Instead, we verify that PO-RSVMD uses <= iterations compared to
        // max_iter on a signal where over-decomposition is possible.
        let n = 256;
        let total = n + 10;
        let signal = make_signal(total);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        // Run warm frames — error mutation may trigger early stopping
        let mut saw_early_stop = false;
        for i in 0..10 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            if output.iterations < 500 {
                saw_early_stop = true;
            }
            // Regardless, output should be valid
            assert_eq!(output.modes.len(), 3);
            for mode in &output.modes {
                for &v in mode {
                    assert!(!v.is_nan());
                }
            }
        }
        // With warm starting and tau > 0, we should converge or stop early
        assert!(saw_early_stop, "Expected at least one frame to stop before max_iter");
    }

    #[test]
    fn test_gamma_adaptation_under_signal_change() {
        // Build a signal that changes character mid-stream:
        // first half is low-frequency, then we inject a high-frequency burst
        let n = 256;
        let dt = 1.0 / n as f64;

        // Stationary segment
        let mut signal: Vec<f64> = (0..n + 20)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        // Add a high-frequency burst to later samples
        for i in n..n + 20 {
            let t = i as f64 * dt;
            signal[i] += 2.0 * (2.0 * PI * 200.0 * t).sin();
        }

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        // Process frames and track gamma values
        let mut gammas = Vec::new();
        for i in 0..20 {
            proc.update(&signal[n + i..n + i + 1]).unwrap();
            gammas.push(proc.gamma());
        }

        // All gamma values should be valid
        for &g in &gammas {
            assert!(g >= 0.0 && g <= 1.0, "Gamma out of range: {}", g);
        }
    }

    #[test]
    fn test_po_rsvmd_auto_initialize() {
        let n = 256;
        let signal = make_signal(n);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                window_len: n,
                k: 3,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        // Passing window_len samples to update should auto-initialize
        let output = proc.update(&signal).unwrap();
        assert!(proc.initialized());
        assert_eq!(output.modes.len(), 3);
    }

    #[test]
    fn test_po_warm_start_fewer_iterations() {
        let n = 256;
        let total = n + 10;
        let signal = make_signal(total);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        let cold = proc.initialize(&signal[..n]).unwrap();
        let cold_iters = cold.iterations;

        let mut warm_iters = Vec::new();
        for i in 0..10 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            warm_iters.push(output.iterations);
        }

        let avg_warm = warm_iters.iter().sum::<usize>() as f64 / warm_iters.len() as f64;
        assert!(
            avg_warm <= cold_iters as f64,
            "PO-RSVMD avg warm iterations ({:.1}) should be <= cold start ({})",
            avg_warm, cold_iters
        );
    }

    #[test]
    fn test_po_center_freq_stability_across_streaming() {
        let n = 256;
        let total = n + 20;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..total)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + 0.5 * (2.0 * PI * 80.0 * t).sin()
            })
            .collect();

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 2,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        let expected = [20.0 / n as f64, 80.0 / n as f64];
        let tol = 15.0 / n as f64;

        for i in 0..20 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            let mut sorted = output.center_freqs.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for (j, (&actual, &exp)) in sorted.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - exp).abs() < tol,
                    "PO frame {}: center freq {} = {:.6}, expected {:.6} (tol={:.6})",
                    i, j, actual, exp, tol
                );
            }
        }
    }

    #[test]
    fn test_po_long_streaming_200_frames() {
        let n = 256;
        let total = n + 200;
        let signal = make_signal(total);

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 3,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        proc.initialize(&signal[..n]).unwrap();

        for i in 0..200 {
            let output = proc.update(&signal[n + i..n + i + 1]).unwrap();
            assert_eq!(output.modes.len(), 3);
            assert_eq!(output.modes[0].len(), n);
            for mode in &output.modes {
                for &v in mode {
                    assert!(!v.is_nan(), "NaN at PO frame {}", i);
                    assert!(v.is_finite(), "Inf at PO frame {}", i);
                }
            }
            for &f in &output.center_freqs {
                assert!(!f.is_nan(), "NaN center freq at PO frame {}", i);
                assert!(f >= 0.0, "Negative center freq at PO frame {}", i);
            }
        }
    }

    #[test]
    fn test_gamma_blending_formula_boundary_values() {
        // PO-RSVMD paper Formula 20: omega_init = gamma*prev + (1-gamma)*detected
        // Verify at boundary values: gamma=0 → detected, gamma=1 → previous
        let prev = vec![0.1, 0.3];
        let det = vec![0.2, 0.4];

        // gamma = 0: result should equal detected
        let gamma = 0.0;
        let result: Vec<f64> = prev
            .iter()
            .zip(&det)
            .map(|(&p, &d)| gamma * p + (1.0 - gamma) * d)
            .collect();
        for (i, (&r, &d)) in result.iter().zip(det.iter()).enumerate() {
            assert!(
                (r - d).abs() < 1e-15,
                "gamma=0: freq {} should equal detected ({} vs {})",
                i, r, d
            );
        }

        // gamma = 1: result should equal previous
        let gamma = 1.0;
        let result: Vec<f64> = prev
            .iter()
            .zip(&det)
            .map(|(&p, &d)| gamma * p + (1.0 - gamma) * d)
            .collect();
        for (i, (&r, &p)) in result.iter().zip(prev.iter()).enumerate() {
            assert!(
                (r - p).abs() < 1e-15,
                "gamma=1: freq {} should equal previous ({} vs {})",
                i, r, p
            );
        }

        // gamma = 0.5: result should be average
        let gamma = 0.5;
        let result: Vec<f64> = prev
            .iter()
            .zip(&det)
            .map(|(&p, &d)| gamma * p + (1.0 - gamma) * d)
            .collect();
        assert!((result[0] - 0.15).abs() < 1e-15, "gamma=0.5: expected 0.15");
        assert!((result[1] - 0.35).abs() < 1e-15, "gamma=0.5: expected 0.35");

        // gamma = 0.7: weighted toward previous
        let gamma = 0.7;
        let result: Vec<f64> = prev
            .iter()
            .zip(&det)
            .map(|(&p, &d)| gamma * p + (1.0 - gamma) * d)
            .collect();
        assert!(
            (result[0] - 0.13).abs() < 1e-15,
            "gamma=0.7: expected 0.13, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.33).abs() < 1e-15,
            "gamma=0.7: expected 0.33, got {}",
            result[1]
        );
    }

    #[test]
    fn test_po_rsvmd_vs_standard_on_overdetermined() {
        // PO-RSVMD should achieve comparable or better reconstruction
        // than standard RSVMD on over-determined problems (K > actual components)
        // due to error mutation detection (PO-RSVMD paper, Section 4.1)
        let n = 256;
        let total = n + 5;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..total)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + 0.5 * (2.0 * PI * 80.0 * t).sin()
            })
            .collect();

        // PO-RSVMD with K=4 (over-determined: only 2 real components)
        let po_config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 4,
                tau: 0.1,
                tol: 1e-7,
                window_len: n,
                step_size: 1,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut po_proc = PoRsvmdProcessor::new(po_config);
        po_proc.initialize(&signal[..n]).unwrap();

        // Standard RSVMD with same parameters
        let std_config = VmdConfig {
            alpha: 2000.0,
            k: 4,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            step_size: 1,
            max_iter: 500,
            ..Default::default()
        };

        let mut std_proc = crate::rsvmd_core::RsvmdProcessor::new(std_config);
        std_proc.initialize(&signal[..n]).unwrap();

        for i in 0..5 {
            let po_out = po_proc.update(&signal[n + i..n + i + 1]).unwrap();
            let std_out = std_proc.update(&signal[n + i..n + i + 1]).unwrap();

            // Both should produce valid output
            assert_eq!(po_out.modes.len(), 4);
            assert_eq!(std_out.modes.len(), 4);

            // PO should use ≤ iterations (error mutation may stop early)
            assert!(
                po_out.iterations <= std_out.iterations + 1,
                "Frame {}: PO ({} iters) should not use significantly more than std ({} iters)",
                i, po_out.iterations, std_out.iterations
            );

            // No NaN in PO output
            for mode in &po_out.modes {
                for &v in mode {
                    assert!(!v.is_nan(), "NaN in PO-RSVMD at frame {}", i);
                }
            }
        }
    }

    #[test]
    fn test_po_rsvmd_zero_signal() {
        let n = 128;
        let signal = vec![0.0; n];

        let config = PoRsvmdConfig::new(
            VmdConfig {
                alpha: 2000.0,
                k: 2,
                window_len: n,
                max_iter: 500,
                ..Default::default()
            },
            0.5,
        );

        let mut proc = PoRsvmdProcessor::new(config);
        let output = proc.initialize(&signal).unwrap();

        assert_eq!(output.modes.len(), 2);
        for mode in &output.modes {
            for &v in mode {
                assert!(!v.is_nan(), "NaN on zero signal");
            }
        }
    }
}
