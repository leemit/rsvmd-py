use num_complex::Complex64;

/// Generate linearly spaced values from start to end (inclusive).
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * step).collect()
}

/// Compute normalized frequency array [0, 1/N, 2/N, ..., (N-1)/N].
pub fn normalized_freqs(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / n as f64).collect()
}

/// Compute the squared L2 norm of a complex slice.
pub fn norm_sqr_sum(v: &[Complex64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum()
}

/// Compute the L2 norm of a complex slice.
pub fn l2_norm(v: &[Complex64]) -> f64 {
    norm_sqr_sum(v).sqrt()
}

/// Pointwise difference squared norm: sum |a[i] - b[i]|^2
pub fn diff_norm_sqr(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm_sqr())
        .sum()
}

/// Reconstruction error: || sum_k modes[k] - signal ||^2
pub fn reconstruction_error(
    modes: &[Vec<Complex64>],
    signal_spectrum: &[Complex64],
) -> f64 {
    let n = signal_spectrum.len();
    let mut err = 0.0;
    for i in 0..n {
        let mut mode_sum = Complex64::new(0.0, 0.0);
        for mode in modes {
            mode_sum += mode[i];
        }
        err += (mode_sum - signal_spectrum[i]).norm_sqr();
    }
    err
}
