use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Recursive sliding DFT processor.
pub struct SlidingDft {
    /// Current DFT bins, length N.
    bins: Vec<Complex64>,
    /// Precomputed twiddle factors: W_N^k = e^{j*2*pi*k/N} for k = 0..N.
    twiddles: Vec<Complex64>,
    /// Window length.
    n: usize,
    /// Damping factor for numerical stability.
    damping: f64,
    /// Precomputed damping^{-N} for damped variant.
    damping_inv_n: f64,
    /// Frame counter for periodic FFT reset.
    frame_count: usize,
    /// How often to reset with full FFT (0 = never).
    fft_reset_interval: usize,
}

impl SlidingDft {
    /// Initialize with full FFT of the first window.
    pub fn new(initial_window: &[f64], damping: f64, fft_reset_interval: usize) -> Self {
        let n = initial_window.len();
        let bins = Self::compute_fft(initial_window);

        // Precompute twiddle factors: W_N^k = e^{j*2*pi*k/N}
        let twiddles: Vec<Complex64> = (0..n)
            .map(|k| {
                let angle = 2.0 * PI * k as f64 / n as f64;
                Complex64::new(angle.cos(), angle.sin())
            })
            .collect();

        let damping_inv_n = damping.powi(-(n as i32));

        SlidingDft {
            bins,
            twiddles,
            n,
            damping,
            damping_inv_n,
            frame_count: 0,
            fft_reset_interval,
        }
    }

    /// Compute full FFT of a real signal.
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

    /// Slide by one sample: O(N) update.
    /// x_new = sample entering the window, x_old = sample leaving the window.
    pub fn slide_one(&mut self, x_new: f64, x_old: f64) {
        for k in 0..self.n {
            let diff = Complex64::new(x_new - self.damping_inv_n * x_old, 0.0);
            self.bins[k] = self.damping * self.twiddles[k] * (self.bins[k] + diff);
        }
        self.frame_count += 1;
    }

    /// Slide by s samples: O(N*s) update via Updating Vector Transform.
    pub fn slide_block(&mut self, new_samples: &[f64], old_samples: &[f64]) -> Result<(), String> {
        let s = new_samples.len();
        if s != old_samples.len() {
            return Err(format!(
                "new_samples length ({}) must match old_samples length ({})",
                s, old_samples.len()
            ));
        }

        if s == 1 {
            self.slide_one(new_samples[0], old_samples[0]);
            return Ok(());
        }

        for k in 0..self.n {
            // Compute UVT: D_s[k] = sum_{j=0}^{s-1} (x_new[j] - x_old[j]) * W_N^{-kj}
            let mut d = Complex64::new(0.0, 0.0);
            let twiddle_conj = self.twiddles[k].conj();
            let mut tw_power = Complex64::new(1.0, 0.0); // W_N^{-k*0} = 1
            for j in 0..s {
                let diff = new_samples[j] - old_samples[j];
                d += Complex64::new(diff, 0.0) * tw_power;
                tw_power *= twiddle_conj;
            }
            // X_{m+s}[k] = W_N^{ks} * (X_m[k] + D_s[k])
            // twiddles[k]^s = W_N^{ks}
            let tw_s = self.twiddle_power(k, s);
            self.bins[k] = tw_s * (self.bins[k] + d);
        }
        self.frame_count += 1;
        Ok(())
    }

    /// Compute twiddles[k]^power efficiently.
    fn twiddle_power(&self, k: usize, power: usize) -> Complex64 {
        let angle = 2.0 * PI * k as f64 * power as f64 / self.n as f64;
        Complex64::new(angle.cos(), angle.sin())
    }

    /// Force full FFT recomputation from window buffer (reset drift).
    pub fn reset_from_buffer(&mut self, window: &[f64]) -> Result<(), String> {
        if window.len() != self.n {
            return Err(format!(
                "Window length ({}) must match N ({})",
                window.len(), self.n
            ));
        }
        self.bins = Self::compute_fft(window);
        self.frame_count = 0;
        Ok(())
    }

    /// Check if periodic reset is due.
    pub fn needs_reset(&self) -> bool {
        self.fft_reset_interval > 0 && self.frame_count > 0
            && self.frame_count % self.fft_reset_interval == 0
    }

    /// Get current spectrum (read-only).
    pub fn spectrum(&self) -> &[Complex64] {
        &self.bins
    }

    /// Get mutable spectrum (for tests/direct manipulation).
    pub fn spectrum_mut(&mut self) -> &mut [Complex64] {
        &mut self.bins
    }

    /// Get window length.
    pub fn len(&self) -> usize {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_signal(n: usize) -> Vec<f64> {
        let dt = 1.0 / n as f64;
        (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_slide_one_matches_full_fft() {
        let n = 256;
        // Create initial window and extended signal
        let signal: Vec<f64> = make_test_signal(n + 10);
        let initial = &signal[0..n];
        let mut sdft = SlidingDft::new(initial, 1.0, 0);

        // Slide one sample
        sdft.slide_one(signal[n], signal[0]);

        // Compute full FFT of shifted window
        let shifted_window = &signal[1..n + 1];
        let expected = SlidingDft::compute_fft(shifted_window);

        // Compare
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            assert!(
                diff < 1e-8,
                "Bin {} differs: sdft={}, expected={}, diff={}",
                k, sdft.bins[k], expected[k], diff
            );
        }
    }

    #[test]
    fn test_slide_multiple_matches_full_fft() {
        let n = 256;
        let slides = 100;
        let signal: Vec<f64> = make_test_signal(n + slides);
        let initial = &signal[0..n];
        let mut sdft = SlidingDft::new(initial, 1.0, 0);

        // Slide one sample at a time
        for i in 0..slides {
            sdft.slide_one(signal[n + i], signal[i]);
        }

        // Compare to full FFT
        let expected = SlidingDft::compute_fft(&signal[slides..slides + n]);
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            assert!(
                diff < 1e-6,
                "Bin {} differs after {} slides: diff={}",
                k, slides, diff
            );
        }
    }

    #[test]
    fn test_block_slide_matches_individual_slides() {
        let n = 128;
        let s = 10;
        let signal: Vec<f64> = make_test_signal(n + s);
        let initial = &signal[0..n];

        // Individual slides
        let mut sdft_individual = SlidingDft::new(initial, 1.0, 0);
        for j in 0..s {
            sdft_individual.slide_one(signal[n + j], signal[j]);
        }

        // Block slide
        let mut sdft_block = SlidingDft::new(initial, 1.0, 0);
        sdft_block.slide_block(&signal[n..n + s], &signal[0..s]).unwrap();

        // Compare
        for k in 0..n {
            let diff = (sdft_individual.bins[k] - sdft_block.bins[k]).norm();
            assert!(
                diff < 1e-8,
                "Bin {} differs between individual and block slide: diff={}",
                k, diff
            );
        }
    }

    #[test]
    fn test_damping_limits_drift() {
        let n = 256;
        let slides = 5000;
        let signal: Vec<f64> = make_test_signal(n + slides);
        let initial = &signal[0..n];
        let mut sdft = SlidingDft::new(initial, 0.99999, 0);

        for i in 0..slides {
            sdft.slide_one(signal[n + i], signal[i]);
        }

        // Compare to full FFT - with damping, should still be reasonably close
        let expected = SlidingDft::compute_fft(&signal[slides..slides + n]);
        let mut max_diff = 0.0f64;
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            max_diff = max_diff.max(diff);
        }
        // Damped SDFT won't match exactly but should be bounded
        // The damping introduces a systematic attenuation, so we allow larger tolerance
        assert!(
            max_diff < 100.0,
            "Max drift after {} slides with damping: {}",
            slides, max_diff
        );
    }

    #[test]
    fn test_reset_from_buffer() {
        let n = 128;
        let signal: Vec<f64> = make_test_signal(n + 50);
        let initial = &signal[0..n];
        let mut sdft = SlidingDft::new(initial, 1.0, 0);

        // Slide some
        for i in 0..50 {
            sdft.slide_one(signal[n + i], signal[i]);
        }

        // Reset
        let new_window = &signal[50..50 + n];
        sdft.reset_from_buffer(new_window).unwrap();

        // Should match full FFT exactly
        let expected = SlidingDft::compute_fft(new_window);
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            assert!(diff < 1e-10, "Bin {} differs after reset: diff={}", k, diff);
        }
    }

    #[test]
    fn test_slide_block_mismatched_lengths_returns_err() {
        let n = 128;
        let signal = make_test_signal(n);
        let mut sdft = SlidingDft::new(&signal, 1.0, 0);

        let result = sdft.slide_block(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must match"));
    }

    #[test]
    fn test_reset_from_buffer_wrong_length_returns_err() {
        let n = 128;
        let signal = make_test_signal(n);
        let mut sdft = SlidingDft::new(&signal, 1.0, 0);

        let result = sdft.reset_from_buffer(&[1.0; 64]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must match N"));
    }

    #[test]
    fn test_block_slide_step5_matches_full_fft() {
        let n = 256;
        let s = 5;
        let signal = make_test_signal(n + s);
        let mut sdft = SlidingDft::new(&signal[..n], 1.0, 0);

        sdft.slide_block(&signal[n..n + s], &signal[0..s]).unwrap();

        let expected = SlidingDft::compute_fft(&signal[s..s + n]);
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            assert!(
                diff < 1e-8,
                "Bin {} differs for step_size=5: diff={}",
                k, diff
            );
        }
    }

    #[test]
    fn test_periodic_fft_reset_fires() {
        let n = 64;
        let signal = make_test_signal(n + 20);
        let mut sdft = SlidingDft::new(&signal[..n], 0.99999, 5);

        // Slide 4 times — no reset yet
        for i in 0..4 {
            sdft.slide_one(signal[n + i], signal[i]);
        }
        assert!(!sdft.needs_reset());

        // 5th slide triggers reset
        sdft.slide_one(signal[n + 4], signal[4]);
        assert!(sdft.needs_reset());
    }

    #[test]
    fn test_damped_sdft_reset_recovers_exact() {
        let n = 128;
        let signal = make_test_signal(n + 100);
        let mut sdft = SlidingDft::new(&signal[..n], 0.99999, 0);

        // Slide with damping — introduces drift
        for i in 0..100 {
            sdft.slide_one(signal[n + i], signal[i]);
        }

        // Reset should recover exact spectrum
        let window = &signal[100..100 + n];
        sdft.reset_from_buffer(window).unwrap();

        let expected = SlidingDft::compute_fft(window);
        for k in 0..n {
            let diff = (sdft.bins[k] - expected[k]).norm();
            assert!(diff < 1e-10, "Bin {} not exact after reset: diff={}", k, diff);
        }
    }
}
