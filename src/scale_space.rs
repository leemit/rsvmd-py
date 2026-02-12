/// Scale-space peak picker for robust initialization of center frequencies.
pub struct ScaleSpacePeakPicker {
    /// Number of smoothing scales.
    n_scales: usize,
    /// Minimum sigma (Gaussian width).
    sigma_min: f64,
    /// Maximum sigma.
    sigma_max: f64,
}

impl ScaleSpacePeakPicker {
    pub fn new(n_scales: usize, sigma_min: f64, sigma_max: f64) -> Self {
        ScaleSpacePeakPicker {
            n_scales,
            sigma_min,
            sigma_max,
        }
    }

    /// Find K most persistent peaks in the power spectrum.
    /// Returns normalized frequencies in [0, 0.5] (positive frequencies only).
    pub fn pick_peaks(&self, power_spectrum: &[f64], k: usize) -> Vec<f64> {
        let n = power_spectrum.len();
        if n < 3 || k == 0 {
            return vec![0.0; k];
        }

        let half_n = n / 2;
        // Only consider positive frequencies
        let positive_spectrum: Vec<f64> = power_spectrum[0..=half_n].to_vec();
        let pn = positive_spectrum.len();

        let mut scores = vec![0.0f64; pn];

        let sigmas = if self.n_scales <= 1 {
            vec![self.sigma_min]
        } else {
            (0..self.n_scales)
                .map(|i| {
                    self.sigma_min
                        + (self.sigma_max - self.sigma_min) * i as f64
                            / (self.n_scales - 1) as f64
                })
                .collect()
        };

        for sigma in &sigmas {
            let smoothed = gaussian_smooth(&positive_spectrum, *sigma);

            // Find local maxima
            for i in 1..pn - 1 {
                if smoothed[i] > smoothed[i - 1] && smoothed[i] > smoothed[i + 1] {
                    scores[i] += sigma; // persistence weighted by scale
                }
            }
        }

        // Return top-K peaks by persistence score
        let mut peak_indices: Vec<usize> = (1..pn - 1)
            .filter(|&i| scores[i] > 0.0)
            .collect();
        peak_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let mut result: Vec<f64> = peak_indices
            .iter()
            .take(k)
            .map(|&i| i as f64 / n as f64) // normalized frequency
            .collect();

        // If we found fewer peaks than K, fill with uniform spacing
        while result.len() < k {
            let idx = result.len();
            result.push((2 * idx + 1) as f64 / (2.0 * k as f64) * 0.5);
        }

        // Sort by frequency
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result
    }
}

/// Apply Gaussian smoothing to a 1D signal.
fn gaussian_smooth(signal: &[f64], sigma: f64) -> Vec<f64> {
    let n = signal.len();
    if sigma < 0.5 {
        return signal.to_vec();
    }

    // Kernel radius: 3*sigma rounded up
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;

    // Build Gaussian kernel
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    // Normalize
    for v in kernel.iter_mut() {
        *v /= sum;
    }

    // Convolve with mirror boundary
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut val = 0.0;
        for j in 0..kernel_size {
            let idx = i as isize + j as isize - radius as isize;
            let idx = mirror_index(idx, n);
            val += signal[idx] * kernel[j];
        }
        result[i] = val;
    }
    result
}

/// Mirror boundary: reflect index into [0, n-1].
fn mirror_index(idx: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = idx;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * (n - 1) - i;
    }
    i.clamp(0, n - 1) as usize
}

/// Default peak picker with reasonable parameters for VMD initialization.
pub fn default_peak_picker() -> ScaleSpacePeakPicker {
    ScaleSpacePeakPicker::new(20, 1.0, 30.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pick_peaks_three_sinusoids() {
        // Create a power spectrum with 3 clear peaks
        let n = 1024;
        let mut spectrum = vec![0.0; n];

        // Place peaks at bins corresponding to normalized frequencies ~0.05, ~0.15, ~0.35
        let peak1 = (0.05 * n as f64) as usize;
        let peak2 = (0.15 * n as f64) as usize;
        let peak3 = (0.35 * n as f64) as usize;

        // Add Gaussian-shaped peaks
        for i in 0..n / 2 + 1 {
            spectrum[i] = 100.0 * (-(i as f64 - peak1 as f64).powi(2) / 20.0).exp()
                + 80.0 * (-(i as f64 - peak2 as f64).powi(2) / 20.0).exp()
                + 60.0 * (-(i as f64 - peak3 as f64).powi(2) / 20.0).exp()
                + 0.1; // noise floor
        }

        let picker = ScaleSpacePeakPicker::new(15, 1.0, 20.0);
        let peaks = picker.pick_peaks(&spectrum, 3);

        assert_eq!(peaks.len(), 3);

        // Check peaks are near the true frequencies (within a few bins)
        let tol = 5.0 / n as f64;
        assert!(
            (peaks[0] - 0.05).abs() < tol,
            "Peak 0: expected ~0.05, got {}",
            peaks[0]
        );
        assert!(
            (peaks[1] - 0.15).abs() < tol,
            "Peak 1: expected ~0.15, got {}",
            peaks[1]
        );
        assert!(
            (peaks[2] - 0.35).abs() < tol,
            "Peak 2: expected ~0.35, got {}",
            peaks[2]
        );
    }

    #[test]
    fn test_pick_peaks_fills_missing() {
        // Spectrum with only 1 clear peak but requesting 3
        let n = 256;
        let mut spectrum = vec![0.1; n];
        spectrum[20] = 100.0;
        spectrum[21] = 80.0;
        spectrum[19] = 80.0;

        let picker = default_peak_picker();
        let peaks = picker.pick_peaks(&spectrum, 3);
        assert_eq!(peaks.len(), 3);
    }

    #[test]
    fn test_peaks_robust_to_noise() {
        // Scale-space should find true persistent peaks despite additive noise
        // (RSVMD paper, Section 3.2)
        let n = 512;
        let peaks = [25usize, 100, 200];
        let mut spectrum = vec![0.1; n];

        // Add Gaussian-shaped peaks
        for &p in &peaks {
            for i in 0..n / 2 + 1 {
                spectrum[i] += 50.0 * (-(i as f64 - p as f64).powi(2) / 30.0).exp();
            }
        }

        // Add deterministic "noise" using golden ratio fractional parts
        for i in 0..n {
            let noise = ((i as f64 * 0.618033988749895).fract() - 0.5) * 8.0;
            spectrum[i] += noise.abs();
        }

        let picker = ScaleSpacePeakPicker::new(20, 1.0, 30.0);
        let found = picker.pick_peaks(&spectrum, 3);
        assert_eq!(found.len(), 3);

        let expected = [25.0 / n as f64, 100.0 / n as f64, 200.0 / n as f64];
        let tol = 5.0 / n as f64;

        for (i, (&f, &e)) in found.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f - e).abs() < tol,
                "Noisy peak {}: expected {:.4}, got {:.4} (tol={:.4})",
                i, e, f, tol
            );
        }
    }

    #[test]
    fn test_gaussian_smooth() {
        let signal = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let smoothed = gaussian_smooth(&signal, 1.0);

        // Peak should be at index 3 but spread out
        assert!(smoothed[3] > smoothed[0]);
        assert!(smoothed[3] > smoothed[2]);
        assert!(smoothed[2] > smoothed[0]);
        // Sum should be approximately preserved
        let sum: f64 = smoothed.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }
}
