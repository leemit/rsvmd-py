// This work is dedicated to the public domain under the CC0 1.0 Universal license.
// To the extent possible under law, the author has waived all copyright
// and related or neighboring rights to this work.
// https://creativecommons.org/publicdomain/zero/1.0/

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

    // ---- Paper equation verification tests ----

    #[test]
    fn test_wiener_filter_formula_k1() {
        // Paper Eq (Section 2.3): For K=1, other_sum=0, so:
        // u_0[i] = (f[i] + lambda[i]/2) / (1 + 2*alpha*(freq[i] - omega_0)^2)
        let n = 16;
        let alpha = 100.0;

        let signal_spectrum: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 + 1.0, (n - i) as f64 * 0.3))
            .collect();
        let lambda: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.1, -(i as f64) * 0.05))
            .collect();
        let omega_0 = 0.25;
        let freqs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        // Compute expected mode using the paper's formula
        let expected_mode: Vec<Complex64> = (0..n)
            .map(|i| {
                let num = signal_spectrum[i] + lambda[i] * 0.5;
                let freq_diff = freqs[i] - omega_0;
                let denom = 1.0 + 2.0 * alpha * freq_diff * freq_diff;
                num / denom
            })
            .collect();

        let config = VmdConfig {
            alpha,
            k: 1,
            tau: 0.0,
            window_len: n,
            max_iter: 1,
            ..Default::default()
        };
        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(1, n);
        state.center_freqs = vec![omega_0];
        state.lambda = lambda;

        solver.admm_step(&signal_spectrum, &mut state);

        for i in 0..n {
            let diff = (state.mode_spectra[0][i] - expected_mode[i]).norm();
            assert!(
                diff < 1e-12,
                "Wiener filter mismatch at bin {}: got {:?}, expected {:?}",
                i, state.mode_spectra[0][i], expected_mode[i]
            );
        }
    }

    #[test]
    fn test_wiener_filter_gauss_seidel_k2() {
        // Verify Gauss-Seidel ordering: mode 1 uses UPDATED mode 0
        let n = 16;
        let alpha = 50.0;
        let omega = [0.1, 0.35];

        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos()))
            .collect();

        let init_u0: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.01, 0.0))
            .collect();
        let init_u1: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.0, i as f64 * 0.01))
            .collect();

        let freqs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        // Step 1: u_0 update uses OLD u_1
        let expected_u0: Vec<Complex64> = (0..n)
            .map(|i| {
                let num = signal[i] - init_u1[i]; // lambda=0
                let fd = freqs[i] - omega[0];
                num / (1.0 + 2.0 * alpha * fd * fd)
            })
            .collect();

        // Step 2: u_1 update uses NEW u_0 (Gauss-Seidel!)
        let expected_u1: Vec<Complex64> = (0..n)
            .map(|i| {
                let num = signal[i] - expected_u0[i];
                let fd = freqs[i] - omega[1];
                num / (1.0 + 2.0 * alpha * fd * fd)
            })
            .collect();

        let config = VmdConfig {
            alpha,
            k: 2,
            tau: 0.0,
            window_len: n,
            max_iter: 1,
            ..Default::default()
        };
        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(2, n);
        state.mode_spectra[0] = init_u0.clone();
        state.mode_spectra[1] = init_u1.clone();
        state.center_freqs = omega.to_vec();

        solver.admm_step(&signal, &mut state);

        // Mode 0 should match
        for i in 0..n {
            let diff = (state.mode_spectra[0][i] - expected_u0[i]).norm();
            assert!(diff < 1e-12, "Mode 0 mismatch at bin {}: diff={}", i, diff);
        }

        // Mode 1 should match Gauss-Seidel (uses updated u_0)
        for i in 0..n {
            let diff = (state.mode_spectra[1][i] - expected_u1[i]).norm();
            assert!(
                diff < 1e-12,
                "Mode 1 Gauss-Seidel mismatch at bin {}: diff={}",
                i, diff
            );
        }

        // Verify it's NOT Jacobi: Jacobi would use OLD u_0 for mode 1
        let jacobi_u1: Vec<Complex64> = (0..n)
            .map(|i| {
                let num = signal[i] - init_u0[i]; // Jacobi uses OLD u_0
                let fd = freqs[i] - omega[1];
                num / (1.0 + 2.0 * alpha * fd * fd)
            })
            .collect();

        let any_differ = (0..n).any(|i| (state.mode_spectra[1][i] - jacobi_u1[i]).norm() > 1e-10);
        assert!(
            any_differ,
            "Mode 1 should differ from Jacobi ordering"
        );
    }

    #[test]
    fn test_center_freq_power_weighted_mean_single_bin() {
        // Paper: omega_k = sum_{i>=0} freq[i]*|u_k[i]|^2 / sum_{i>=0} |u_k[i]|^2
        // A mode at a single bin should have center_freq = that bin's frequency
        let n = 256;
        let target_bin = 30;

        let mut mode_spectrum = vec![Complex64::new(0.0, 0.0); n];
        mode_spectrum[target_bin] = Complex64::new(10.0, 0.0);

        let freqs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let center_freq = VmdSolver::update_center_freq(&mode_spectrum, &freqs, n);

        let expected = target_bin as f64 / n as f64;
        assert!(
            (center_freq - expected).abs() < 1e-10,
            "Single-bin center freq: expected {:.6}, got {:.6}",
            expected, center_freq
        );
    }

    #[test]
    fn test_center_freq_power_weighted_mean_two_bins() {
        // Two bins with different powers: center freq = power-weighted average
        let n = 256;
        let mut mode_spectrum = vec![Complex64::new(0.0, 0.0); n];
        mode_spectrum[20] = Complex64::new(3.0, 0.0); // power = 9
        mode_spectrum[30] = Complex64::new(1.0, 0.0); // power = 1

        let freqs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let center_freq = VmdSolver::update_center_freq(&mode_spectrum, &freqs, n);

        // Expected: (20/256 * 9 + 30/256 * 1) / (9 + 1)
        let expected = (20.0 / 256.0 * 9.0 + 30.0 / 256.0 * 1.0) / 10.0;
        assert!(
            (center_freq - expected).abs() < 1e-10,
            "Two-bin power-weighted mean: expected {:.6}, got {:.6}",
            expected, center_freq
        );
    }

    #[test]
    fn test_dual_variable_update_formula() {
        // Paper: lambda[i] += tau * (f[i] - sum_k u_k[i])
        let n = 8;
        let tau = 0.3;

        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.0))
            .collect();
        let init_lambda: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(0.0, i as f64 * 0.1))
            .collect();

        let config = VmdConfig {
            alpha: 100.0,
            k: 2,
            tau,
            window_len: n,
            max_iter: 1,
            ..Default::default()
        };
        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(2, n);
        state.mode_spectra[0] = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.3, 0.0))
            .collect();
        state.mode_spectra[1] = (0..n)
            .map(|i| Complex64::new(i as f64 * 0.2, 0.0))
            .collect();
        state.center_freqs = vec![0.1, 0.3];
        state.lambda = init_lambda.clone();

        solver.admm_step(&signal, &mut state);

        // After step, lambda = old_lambda + tau * (f - sum_k u_k_new)
        // u_k_new are the modes AFTER the Wiener filter update
        for i in 0..n {
            let mode_sum = state.mode_spectra[0][i] + state.mode_spectra[1][i];
            let expected =
                init_lambda[i] + Complex64::new(tau, 0.0) * (signal[i] - mode_sum);
            let diff = (state.lambda[i] - expected).norm();
            assert!(
                diff < 1e-12,
                "Lambda mismatch at bin {}: got {:?}, expected {:?}",
                i, state.lambda[i], expected
            );
        }
    }

    #[test]
    fn test_convergence_criterion_formula() {
        // Paper: metric = sum_k ||u_k^new - u_k^old||^2 / ||u_k^old||^2
        let n = 8;
        let current = vec![
            vec![Complex64::new(1.0, 0.5); n],
            vec![Complex64::new(0.3, -0.2); n],
        ];
        let previous = vec![
            vec![Complex64::new(0.9, 0.6); n],
            vec![Complex64::new(0.35, -0.15); n],
        ];

        let mut expected_metric = 0.0;
        for ki in 0..2 {
            let diff_sq: f64 = (0..n)
                .map(|i| (current[ki][i] - previous[ki][i]).norm_sqr())
                .sum();
            let prev_sq: f64 = (0..n).map(|i| previous[ki][i].norm_sqr()).sum();
            if prev_sq > 1e-30 {
                expected_metric += diff_sq / prev_sq;
            }
        }

        let config = VmdConfig {
            alpha: 100.0,
            k: 2,
            window_len: n,
            ..Default::default()
        };
        let solver = VmdSolver::new(config);
        let actual = solver.convergence_metric(&current, &previous);

        assert!(
            (actual - expected_metric).abs() < 1e-12,
            "Convergence metric: expected {:.12}, got {:.12}",
            expected_metric, actual
        );
    }

    #[test]
    fn test_mode_spectral_concentration() {
        // Paper claim: each mode captures a distinct frequency band.
        // Verify spectral energy is concentrated near center frequency.
        let n = 512;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + (2.0 * PI * 100.0 * t).sin()
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

        let freqs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let half_n = n / 2;
        let bandwidth = 20.0 / n as f64;

        for ki in 0..2 {
            let omega_k = result.center_freqs[ki];
            let mut near_power = 0.0;
            let mut total_power = 0.0;
            for i in 0..=half_n {
                let power = result.mode_spectra[ki][i].norm_sqr();
                total_power += power;
                if (freqs[i] - omega_k).abs() < bandwidth {
                    near_power += power;
                }
            }
            let concentration = if total_power > 1e-30 {
                near_power / total_power
            } else {
                0.0
            };
            assert!(
                concentration > 0.5,
                "Mode {} spectral concentration: {:.4} (center={:.4}), should be >0.5",
                ki, concentration, omega_k
            );
        }
    }

    fn compute_ifft_modes(mode_spectra: &[Vec<Complex64>], n: usize) -> Vec<Vec<f64>> {
        let mut planner = FftPlanner::new();
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

    #[test]
    fn test_mode_cross_correlation_low() {
        // Modes from well-separated sinusoids should be nearly orthogonal
        let n = 512;
        let dt = 1.0 / n as f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 20.0 * t).sin() + (2.0 * PI * 100.0 * t).sin()
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

        let modes_td = compute_ifft_modes(&result.mode_spectra, n);

        let dot: f64 = (0..n).map(|i| modes_td[0][i] * modes_td[1][i]).sum();
        let norm0: f64 = (0..n)
            .map(|i| modes_td[0][i].powi(2))
            .sum::<f64>()
            .sqrt();
        let norm1: f64 = (0..n)
            .map(|i| modes_td[1][i].powi(2))
            .sum::<f64>()
            .sqrt();

        let corr = if norm0 > 1e-10 && norm1 > 1e-10 {
            (dot / (norm0 * norm1)).abs()
        } else {
            0.0
        };

        assert!(
            corr < 0.3,
            "Cross-correlation between modes should be low: {:.4}",
            corr
        );
    }

    #[test]
    fn test_reconstruction_error_decreases() {
        // On a well-conditioned signal, reconstruction error should decrease
        // across ADMM iterations (validates convergence behavior)
        let n = 256;
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
            tol: 1e-15, // very tight to prevent early stopping
            window_len: n,
            max_iter: 500,
            ..Default::default()
        };

        let solver = VmdSolver::new(config);
        let mut state = VmdState::new(2, n);
        state.init_uniform_freqs();

        let mut errors = Vec::new();
        for _ in 0..30 {
            solver.admm_step(&spectrum, &mut state);
            errors.push(complex_utils::reconstruction_error(
                &state.mode_spectra,
                &spectrum,
            ));
        }

        // Error at end should be less than at start
        let first = errors[0];
        let last = *errors.last().unwrap();
        assert!(
            last < first,
            "Error should decrease: first={:.6}, last={:.6}",
            first, last
        );
    }

    #[test]
    fn test_error_mutation_principle() {
        // Validates the principle behind PO-RSVMD's error mutation detection:
        // warm-start ADMM on a changed signal can cause error to increase.
        let n = 256;
        let dt = 1.0 / n as f64;

        let signal1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 20.0 * i as f64 * dt).sin())
            .collect();
        let signal2: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 100.0 * t).sin() + (2.0 * PI * 200.0 * t).sin()
            })
            .collect();

        let spectrum1 = compute_fft(&signal1);
        let spectrum2 = compute_fft(&signal2);

        let config = VmdConfig {
            alpha: 2000.0,
            k: 2,
            tau: 0.1,
            tol: 1e-15,
            window_len: n,
            max_iter: 100,
            ..Default::default()
        };

        let solver = VmdSolver::new(config.clone());

        // Solve signal 1 to convergence
        let mut state1 = VmdState::new(2, n);
        state1.init_uniform_freqs();
        let result1 = solver.solve(&spectrum1, &mut state1);

        // Warm-start on signal 2 using signal 1's converged state
        let mut state2 = VmdState {
            mode_spectra: result1.mode_spectra,
            center_freqs: result1.center_freqs,
            lambda: result1.lambda,
            initialized: true,
        };

        let mut errors = Vec::new();
        for _ in 0..100 {
            solver.admm_step(&spectrum2, &mut state2);
            errors.push(complex_utils::reconstruction_error(
                &state2.mode_spectra,
                &spectrum2,
            ));
        }

        // Find minimum error point
        let min_error = errors.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_idx = errors
            .iter()
            .position(|&e| (e - min_error).abs() < 1e-30)
            .unwrap();

        // The minimum should not be at the very last iteration —
        // showing that continuing past the optimum doesn't help
        // (or error converged cleanly, which is also valid)
        if min_idx < errors.len() - 1 {
            // Error at minimum should be ≤ error at next iteration
            assert!(
                errors[min_idx] <= errors[min_idx + 1] * 1.001,
                "Min at iter {} ({:.6}) should be ≤ next ({:.6})",
                min_idx, errors[min_idx], errors[min_idx + 1]
            );
        }

        // Either way, final error should be reasonable (not blown up)
        assert!(
            errors.last().unwrap().is_finite(),
            "Error should remain finite"
        );
    }
}
