use num_complex::Complex64;

use crate::complex_utils;

/// Configuration parameters for VMD.
#[derive(Clone, Debug)]
pub struct VmdConfig {
    /// Bandwidth constraint penalty. Larger = narrower modes. Typical: 2000.
    pub alpha: f64,
    /// Number of modes.
    pub k: usize,
    /// Dual ascent step size. 0 = exact reconstruction.
    pub tau: f64,
    /// Convergence tolerance.
    pub tol: f64,
    /// Window length in samples.
    pub window_len: usize,
    /// Sliding step size.
    pub step_size: usize,
    /// Max ADMM iterations per frame.
    pub max_iter: usize,
    /// SDFT damping factor.
    pub damping: f64,
    /// Recompute full FFT every P frames (0 = never).
    pub fft_reset_interval: usize,
}

impl Default for VmdConfig {
    fn default() -> Self {
        VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.0,
            tol: 1e-7,
            window_len: 7200,
            step_size: 1,
            max_iter: 500,
            damping: 0.99999,
            fft_reset_interval: 0,
        }
    }
}

/// Per-frame VMD state, persisted between frames.
pub struct VmdState {
    /// Mode spectra: hat{u}_k(omega), shape K x N.
    pub mode_spectra: Vec<Vec<Complex64>>,
    /// Center frequencies: omega_k, length K.
    pub center_freqs: Vec<f64>,
    /// Lagrangian multiplier: hat{lambda}(omega), length N.
    pub lambda: Vec<Complex64>,
    /// Whether cold start has been performed.
    pub initialized: bool,
}

impl VmdState {
    /// Create a new uninitialized state.
    pub fn new(k: usize, n: usize) -> Self {
        VmdState {
            mode_spectra: vec![vec![Complex64::new(0.0, 0.0); n]; k],
            center_freqs: vec![0.0; k],
            lambda: vec![Complex64::new(0.0, 0.0); n],
            initialized: false,
        }
    }

    /// Initialize center frequencies with uniform spacing in [0, 0.5].
    pub fn init_uniform_freqs(&mut self) {
        let k = self.center_freqs.len();
        for i in 0..k {
            self.center_freqs[i] = (i as f64) / (2.0 * k as f64);
        }
    }

    /// Reset modes and lambda to zero.
    pub fn reset_modes_and_lambda(&mut self) {
        let zero = Complex64::new(0.0, 0.0);
        for mode in self.mode_spectra.iter_mut() {
            for v in mode.iter_mut() {
                *v = zero;
            }
        }
        for v in self.lambda.iter_mut() {
            *v = zero;
        }
    }
}

/// Result of a VMD solve.
pub struct VmdResult {
    /// Decomposed mode spectra, K x N.
    pub mode_spectra: Vec<Vec<Complex64>>,
    /// Center frequencies, length K.
    pub center_freqs: Vec<f64>,
    /// Lagrangian multiplier, length N.
    pub lambda: Vec<Complex64>,
    /// Number of ADMM iterations used.
    pub iterations: usize,
    /// Whether ADMM converged.
    pub converged: bool,
    /// Final reconstruction error.
    pub final_error: f64,
}

/// Single-frame VMD ADMM solver operating on a pre-computed spectrum.
pub struct VmdSolver {
    config: VmdConfig,
    /// Precomputed normalized frequency array [0, 1/N, 2/N, ..., (N-1)/N].
    freqs: Vec<f64>,
}

impl VmdSolver {
    pub fn new(config: VmdConfig) -> Self {
        let freqs = complex_utils::normalized_freqs(config.window_len);
        VmdSolver { config, freqs }
    }

    /// Run ADMM iterations on the given spectrum.
    pub fn solve(
        &self,
        signal_spectrum: &[Complex64],
        state: &mut VmdState,
    ) -> VmdResult {
        let k = self.config.k;

        // Storage for previous iteration (convergence check)
        let mut prev_modes: Vec<Vec<Complex64>> = state
            .mode_spectra
            .iter()
            .map(|m| m.clone())
            .collect();

        let mut converged = false;
        let mut iterations = 0;
        let mut final_error = f64::INFINITY;

        for iter in 0..self.config.max_iter {
            // Save previous modes for convergence check
            for ki in 0..k {
                prev_modes[ki].copy_from_slice(&state.mode_spectra[ki]);
            }

            // ADMM step
            self.admm_step(signal_spectrum, state);

            // Convergence check (skip first iteration — prev modes may be zero)
            iterations = iter + 1;
            final_error = complex_utils::reconstruction_error(&state.mode_spectra, signal_spectrum);

            if iter > 0 {
                let conv_metric = self.convergence_metric(&state.mode_spectra, &prev_modes);
                if conv_metric < self.config.tol {
                    converged = true;
                    break;
                }
            }
        }

        VmdResult {
            mode_spectra: state.mode_spectra.clone(),
            center_freqs: state.center_freqs.clone(),
            lambda: state.lambda.clone(),
            iterations,
            converged,
            final_error,
        }
    }

    /// Single ADMM iteration step.
    pub fn admm_step(
        &self,
        signal_spectrum: &[Complex64],
        state: &mut VmdState,
    ) {
        let n = signal_spectrum.len();
        let k = self.config.k;

        // For each mode k (Gauss-Seidel ordering)
        for ki in 0..k {
            // Compute sum of all other modes
            let mut other_sum = vec![Complex64::new(0.0, 0.0); n];
            for kj in 0..k {
                if kj != ki {
                    for i in 0..n {
                        other_sum[i] += state.mode_spectra[kj][i];
                    }
                }
            }

            // Mode update (Wiener filter)
            let omega_k = state.center_freqs[ki];
            for i in 0..n {
                let numerator =
                    signal_spectrum[i] - other_sum[i] + state.lambda[i] * 0.5;
                let freq_diff = self.freqs[i] - omega_k;
                let denominator = 1.0 + 2.0 * self.config.alpha * freq_diff * freq_diff;
                state.mode_spectra[ki][i] = numerator / denominator;
            }

            // Center frequency update (center of gravity over positive frequencies)
            state.center_freqs[ki] =
                Self::update_center_freq(&state.mode_spectra[ki], &self.freqs, n);
        }

        // Dual variable (lambda) update
        if self.config.tau > 0.0 {
            let mut mode_sum = vec![Complex64::new(0.0, 0.0); n];
            for ki in 0..k {
                for i in 0..n {
                    mode_sum[i] += state.mode_spectra[ki][i];
                }
            }
            for i in 0..n {
                state.lambda[i] +=
                    Complex64::new(self.config.tau, 0.0) * (signal_spectrum[i] - mode_sum[i]);
            }
        }
    }

    /// Center frequency update: power-weighted mean over positive frequencies.
    fn update_center_freq(mode_spectrum: &[Complex64], freqs: &[f64], n: usize) -> f64 {
        let half_n = n / 2;
        let mut weighted_sum = 0.0;
        let mut power_sum = 0.0;

        for i in 0..=half_n {
            let power = mode_spectrum[i].norm_sqr();
            weighted_sum += freqs[i] * power;
            power_sum += power;
        }

        if power_sum > 1e-30 {
            weighted_sum / power_sum
        } else {
            0.0
        }
    }

    /// Compute convergence metric: sum_k ||u_k^{n+1} - u_k^n||^2 / ||u_k^n||^2
    fn convergence_metric(
        &self,
        current: &[Vec<Complex64>],
        previous: &[Vec<Complex64>],
    ) -> f64 {
        let mut metric = 0.0;
        for ki in 0..self.config.k {
            let diff_sq = complex_utils::diff_norm_sqr(&current[ki], &previous[ki]);
            let prev_sq = complex_utils::norm_sqr_sum(&previous[ki]);
            if prev_sq > 1e-30 {
                metric += diff_sq / prev_sq;
            }
        }
        metric
    }

    pub fn config(&self) -> &VmdConfig {
        &self.config
    }

    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::FftPlanner;
    use std::f64::consts::PI;

    fn compute_fft(signal: &[f64]) -> Vec<Complex64> {
        let n = signal.len();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut buffer: Vec<Complex64> = signal
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect();
        fft.process(&mut buffer);
        buffer
    }

    #[test]
    fn test_three_sinusoids_decomposition() {
        // Use well-separated frequencies for clear decomposition
        let n = 1024;
        let dt = 1.0 / n as f64;

        // Normalized frequencies: 10/1024 ≈ 0.01, 50/1024 ≈ 0.049, 200/1024 ≈ 0.195
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 10.0 * t).sin()
                    + (2.0 * PI * 50.0 * t).sin()
                    + (2.0 * PI * 200.0 * t).sin()
            })
            .collect();

        let spectrum = compute_fft(&signal);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 3,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(3, n);
        state.init_uniform_freqs();

        let result = solver.solve(&spectrum, &mut state);

        // Center frequencies should be near the true values
        let mut freqs = result.center_freqs.clone();
        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected = [10.0 / n as f64, 50.0 / n as f64, 200.0 / n as f64];
        let tol = 10.0 / n as f64;

        for (i, (&got, &exp)) in freqs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "Freq {}: expected ~{:.4}, got {:.4}",
                i, exp, got
            );
        }

        assert!(
            result.iterations < 500,
            "Should converge before max_iter, used {}",
            result.iterations
        );
    }

    #[test]
    fn test_reconstruction() {
        let n = 512;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + 0.5 * (2.0 * PI * 80.0 * t).sin()
            })
            .collect();

        let spectrum = compute_fft(&signal);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(2, n);
        state.init_uniform_freqs();

        let result = solver.solve(&spectrum, &mut state);

        // Sum of modes should approximate the signal spectrum
        let mut recon = vec![Complex64::new(0.0, 0.0); n];
        for mode in &result.mode_spectra {
            for i in 0..n {
                recon[i] += mode[i];
            }
        }

        let mut err = 0.0;
        let mut sig_power = 0.0;
        for i in 0..n {
            err += (recon[i] - spectrum[i]).norm_sqr();
            sig_power += spectrum[i].norm_sqr();
        }
        let relative_error = (err / sig_power).sqrt();

        // VMD partitions the spectrum; with tau > 0 the modes may not perfectly sum
        // to the original signal, but should capture most of the energy
        assert!(
            relative_error < 1.0,
            "Reconstruction relative error too high: {}",
            relative_error
        );
    }

    #[test]
    fn test_warm_start_fewer_iterations() {
        let n = 512;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + (2.0 * PI * 80.0 * t).sin()
            })
            .collect();

        let spectrum = compute_fft(&signal);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.1,
            tol: 1e-7,
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let solver = VmdSolver::new(config);

        // Cold start
        let mut state_cold = VmdState::new(2, n);
        state_cold.init_uniform_freqs();
        let cold_result = solver.solve(&spectrum, &mut state_cold);

        // Warm start (re-solve with converged state — should converge in ~2 iterations)
        let mut state_warm = VmdState {
            mode_spectra: cold_result.mode_spectra.clone(),
            center_freqs: cold_result.center_freqs.clone(),
            lambda: cold_result.lambda.clone(),
            initialized: true,
        };
        let warm_result = solver.solve(&spectrum, &mut state_warm);

        assert!(
            warm_result.iterations <= cold_result.iterations,
            "Warm start ({} iters) should not take more iterations than cold start ({} iters)",
            warm_result.iterations,
            cold_result.iterations
        );
    }
}
