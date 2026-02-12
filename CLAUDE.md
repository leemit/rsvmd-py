# rsvmd-py: Recursive Sliding Variational Mode Decomposition

A Rust implementation of RSVMD and PO-RSVMD with Python bindings via PyO3, designed for real-time signal decomposition in streaming/sliding-window applications.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Standard VMD Foundation](#2-standard-vmd-foundation)
3. [RSVMD Algorithm](#3-rsvmd-algorithm)
4. [PO-RSVMD Algorithm](#4-po-rsvmd-algorithm)
5. [Rust Architecture](#5-rust-architecture)
6. [Python Bindings](#6-python-bindings)
7. [Project Scaffolding](#7-project-scaffolding)
8. [Test Plan](#8-test-plan)
9. [Integration Notes (ctrade)](#9-integration-notes-ctrade)
10. [References](#10-references)

---

## 1. Project Overview

### What

RSVMD (Recursive Sliding Variational Mode Decomposition) is a real-time variant of VMD that replaces the batch FFT with a recursive Sliding DFT, warm-starts each frame from the previous frame's results, and uses scale-space peak picking for robust initialization. PO-RSVMD extends this with adaptive center frequency initialization and over-decomposition prevention.

### Why Rust

- The inner loop is a tight numerical computation (complex Wiener filtering per frequency bin per ADMM iteration per frame) competing for time in latency-sensitive pipelines.
- Recursive state (DFT coefficients, center frequencies, mode estimates) persists across calls. Rust's ownership model prevents state corruption.
- PyO3/maturin produces pip-installable wheels with zero Rust toolchain required on the consumer side.

### Target API

```python
from rsvmd import RSVMDProcessor, PORSVMDProcessor

# Standard RSVMD
proc = RSVMDProcessor(
    alpha=2000.0,       # bandwidth constraint
    K=3,                # number of modes
    tau=0.0,            # noise tolerance (0 = exact reconstruction)
    tol=1e-7,           # convergence tolerance
    window_len=7200,    # window length in samples
    step_size=1,        # sliding step (1 = sample-by-sample)
    max_iter=500,       # max ADMM iterations per frame
)

# First call: cold start with full FFT + scale-space peak picking
imfs, center_freqs = proc.update(initial_window)  # shape: (K, window_len), (K,)

# Subsequent calls: recursive sliding DFT, warm-started ADMM
imfs, center_freqs = proc.update(new_samples)  # new_samples length = step_size

# PO-RSVMD variant with adaptive center frequency and over-decomposition prevention
po_proc = PORSVMDProcessor(
    alpha=2000.0,
    K=3,
    tau=0.0,
    tol=1e-7,
    window_len=7200,
    step_size=1,
    max_iter=500,
    gamma_default=0.5,  # default rate learning factor
)

imfs, center_freqs = po_proc.update(new_samples)
```

---

## 2. Standard VMD Foundation

RSVMD modifies standard VMD, so understanding the base algorithm is essential.

### 2.1 Constrained Variational Problem

VMD decomposes a real-valued signal f(t) into K modes {u_k(t)} with center frequencies {omega_k}, minimizing the aggregate bandwidth:

```
min_{{u_k},{omega_k}}  sum_{k=1}^{K} || d/dt [ (delta(t) + j/(pi*t)) * u_k(t) ] * e^{-j*omega_k*t} ||_2^2

subject to:  sum_{k=1}^{K} u_k(t) = f(t)
```

For each mode u_k:
1. Compute the analytic signal via Hilbert transform: `(delta(t) + j/(pi*t)) * u_k(t)`
2. Shift to baseband by mixing with `e^{-j*omega_k*t}`
3. Estimate bandwidth as the squared L2 norm of the time derivative (gradient)

### 2.2 Augmented Lagrangian

```
L({u_k}, {omega_k}, lambda) =
    alpha * sum_k || d/dt [ (delta(t) + j/(pi*t)) * u_k(t) ] * e^{-j*omega_k*t} ||_2^2
  + || f(t) - sum_k u_k(t) ||_2^2
  + <lambda(t),  f(t) - sum_k u_k(t)>
```

Parameters:
- `alpha` — bandwidth constraint penalty. Larger = narrower modes. Typical: 2000.
- `lambda(t)` — Lagrangian multiplier enforcing reconstruction.
- `tau` — dual ascent step size. 0 = no noise tolerance. >0 allows reconstruction error for noisy signals.

### 2.3 ADMM Update Equations (Fourier Domain)

All operations are performed in the frequency domain via Parseval's theorem. Let `hat{x}(omega)` denote the Fourier transform of `x(t)`.

**Mode update (Wiener filter) — for each k = 1, ..., K:**

```
hat{u}_k^{n+1}(omega) = [ hat{f}(omega) - sum_{i != k} hat{u}_i(omega) + hat{lambda}^n(omega) / 2 ]
                         / [ 1 + 2 * alpha * (omega - omega_k^n)^2 ]
```

Interpretation: bandpass Wiener filter centered at omega_k. The denominator is a Lorentzian with Q-factor controlled by alpha. The sum uses Gauss-Seidel ordering: hat{u}_i^{n+1} for i < k (already updated), hat{u}_i^{n} for i > k.

**Center frequency update — for each k:**

```
omega_k^{n+1} = sum_{omega >= 0} omega * |hat{u}_k^{n+1}(omega)|^2
              / sum_{omega >= 0} |hat{u}_k^{n+1}(omega)|^2
```

Interpretation: center of gravity (power-weighted mean) of the mode's power spectrum over non-negative frequencies.

**Dual variable update:**

```
hat{lambda}^{n+1}(omega) = hat{lambda}^n(omega) + tau * ( hat{f}(omega) - sum_k hat{u}_k^{n+1}(omega) )
```

### 2.4 Convergence Criterion

```
sum_k || hat{u}_k^{n+1} - hat{u}_k^n ||_2^2  /  || hat{u}_k^n ||_2^2  <  tol
```

Typical tol = 1e-7. Also impose max_iter as safety bound.

### 2.5 Discrete Implementation Notes

For a discrete signal of length N:
- Frequency bins: omega_l = 2*pi*l/N for l = 0, 1, ..., N-1
- Only positive frequencies (l = 0 to N/2) are needed due to conjugate symmetry of real signals
- All mode/frequency/lambda updates are pointwise per frequency bin — no matrix inversions
- Signal mirroring at boundaries (extend to 2N by symmetric reflection) reduces Gibbs artifacts
- Complexity per iteration: O(K*N) after the initial O(N log N) FFT

### 2.6 Initialization

- Center frequencies: uniform spacing `omega_k^0 = (k-1) / (2K)` in normalized frequency [0, 0.5]
- Modes: hat{u}_k = 0 for all k
- Dual variable: hat{lambda} = 0
- VMD is largely robust to initialization — center frequencies self-organize toward spectral peaks

---

## 3. RSVMD Algorithm

### 3.1 Core Modification: Sliding DFT

Instead of computing a full FFT each frame, RSVMD recursively updates the DFT as the window slides.

**Single-sample slide:**

```
X_{m+1}[k] = W_N^k * ( X_m[k] + x[m+N] - x[m] )
```

where:
- `X_m[k]` = k-th frequency bin of DFT at window position m
- `W_N = e^{j*2*pi/N}` (twiddle factor)
- `x[m+N]` = new sample entering the window
- `x[m]` = old sample leaving the window

**Block slide with step size s:**

```
X_{m+s}[k] = W_N^{ks} * ( X_m[k] + D_s[k] )
```

where `D_s[k]` is the Updating Vector Transform (UVT):

```
D_s[k] = sum_{j=0}^{s-1} [ x[m+N+j] - x[m+j] ] * W_N^{-kj}
```

**Complexity savings**: O(N) per single-sample slide vs O(N log N) for full FFT. For block slide with step s: O(N*s).

**Numerical stability**: Recursive SDFT accumulates rounding error. Apply a damping factor r (e.g., 0.99999):

```
X_{m+1}[k] = r * W_N^k * ( X_m[k] + x[m+N] - r^{-N} * x[m] )
```

Alternatively, recompute a full FFT every P frames (e.g., P=100) to reset drift.

### 3.2 Scale-Space Peak Picking (First Frame Initialization)

Used only on the first frame (cold start) to determine K initial center frequencies from the power spectrum.

**Algorithm:**
1. Compute power spectrum: `S[k] = |hat{f}[k]|^2`
2. For sigma = sigma_min to sigma_max (increasing Gaussian kernel widths):
   a. Smooth: `S_sigma = GaussianFilter(S, sigma)`
   b. Find local maxima of S_sigma
   c. Associate each maximum to the nearest maximum from the previous scale
   d. Increment persistence score for maxima that survive this scale
3. Select K peaks with highest persistence scores
4. Their frequency positions become the initial center frequencies

This is more robust than simple peak detection because transient noise peaks are eliminated by smoothing while true spectral peaks persist across scales.

### 3.3 Warm-Starting

For frames m >= 1:
- Center frequencies: `omega_k^{init}(m) = omega_k^{final}(m-1)`
- Modes: `hat{u}_k^{init}(m) = hat{u}_k^{final}(m-1)`
- Result: convergence typically requires 2-5 ADMM iterations (vs 50-500 cold-start)

### 3.4 Boundary Effect Handling (Displacement Technology)

At window edges, the decomposition quality degrades. RSVMD handles this by:
1. Only outputting the central portion of each frame's decomposition
2. Replacing edge samples with overlapping results from the previous frame's decomposition
3. The overlap region (N - s samples) provides redundancy for smooth transitions

### 3.5 Tightened Convergence

RSVMD uses the same criterion as standard VMD but with a tighter epsilon, because warm-starting means the iteration begins close to the solution. Both absolute and relative tolerances should be met:

```
Absolute:  || hat{u}_k^{n+1} - hat{u}_k^n ||_2  <  tol_abs
Relative:  || hat{u}_k^{n+1} - hat{u}_k^n ||_2  /  || hat{u}_k^n ||_2  <  tol_rel
```

### 3.6 Complete RSVMD Pseudocode

```
COLD START (Frame 0):
  1. Collect first N samples
  2. Compute hat{f} = FFT(x[0:N])
  3. Run scale-space peak picking on |hat{f}|^2 to get omega_k^0
  4. Initialize hat{u}_k = 0, hat{lambda} = 0
  5. Run ADMM loop until convergence (standard VMD)
  6. Store hat{f}, omega_k^final, hat{u}_k^final

SUBSEQUENT FRAMES (m = 1, 2, ...):
  1. Receive s new samples
  2. Sliding DFT: for each bin k:
       hat{f}^(m)[k] = W_N^{ks} * (hat{f}^(m-1)[k] + D_s[k])
  3. Initialize: omega_k = omega_k^final(m-1), hat{u}_k = hat{u}_k^final(m-1)
  4. ADMM loop (typically 2-5 iterations):
     For each mode k:
       hat{u}_k = Wiener_filter(hat{f}, other_modes, hat{lambda}, alpha, omega_k)
       omega_k = center_of_gravity(hat{u}_k)
     hat{lambda} += tau * (hat{f} - sum_k hat{u}_k)
     Check convergence (absolute AND relative)
  5. Apply boundary displacement
  6. Output central portion of IFFT(hat{u}_k) for each mode
  7. Store state for next frame
```

---

## 4. PO-RSVMD Algorithm

PO-RSVMD addresses two failure modes of standard RSVMD under strong interference:

### 4.1 Problem 1: Over-Decomposition

After the optimal decomposition is reached, continued ADMM iteration can cause the error to **increase** rather than decrease. Standard RSVMD's convergence criterion doesn't detect this.

**Solution — Error Mutation Detection:**

Monitor the reconstruction error between successive iterations:

```
e^n = || sum_k hat{u}_k^{n} - hat{f} ||_2^2    (or modal component error)
```

If the error increases between iterations:

```
e^{n+1} > e^n    ("error mutation")
```

Then terminate immediately and roll back to iteration n's result. This prevents the decomposition from degrading past its optimal point.

**Implementation:**

```
prev_error = infinity
for n in 0..max_iter:
    // ... ADMM update steps ...

    curr_error = reconstruction_error(modes, signal_spectrum)

    if curr_error > prev_error:
        // Error mutation detected — over-decomposition
        // Roll back to previous iteration's modes
        restore_previous_iteration()
        break

    if converged(tol_abs, tol_rel):
        break

    prev_error = curr_error
```

### 4.2 Problem 2: Center Frequency Drift Under Interference

When the signal changes significantly between frames, directly using the previous frame's center frequencies as initialization can introduce growing error.

**Solution — Adaptive Rate Learning Factor:**

Instead of direct warm-start, blend previous and current estimates:

```
omega_k^{init}(m) = gamma * omega_k^{final}(m-1) + (1 - gamma) * omega_k^{detected}(m)
```

where `gamma` is the rate learning factor (called `a` in the paper), adapted based on iteration time changes.

**Adaptation logic (Formula 20 from the PO-RSVMD paper):**

`delta_t` is the absolute difference between the two most recent iteration times: `delta_t = |iteration_time(m) - iteration_time(m-1)|`. The mapping from `delta_t` to the rate factor `gamma` is a 6-tier piecewise function:

```
gamma = 0.0,    delta_t >= 0.8    // severe instability, ignore previous result entirely
        0.001,  0.6 <= delta_t < 0.8   // very high instability
        0.01,   0.4 <= delta_t < 0.6   // high instability
        0.05,   0.2 <= delta_t < 0.4   // moderate instability
        0.2,    0.1 <= delta_t < 0.2   // mild instability
        0.5,    delta_t < 0.1          // stable (default)
```

The insight: iteration time correlates with how much the signal has changed. Stable iteration times (small delta_t) mean the signal is stable and previous frequencies are trustworthy (higher gamma). Large delta_t indicates the signal has shifted significantly and the previous result should be largely discarded (lower gamma, approaching 0).

The `omega_k^{detected}(m)` for the current frame can be obtained by running scale-space peak picking on the current frame's spectrum, or by running a few ADMM iterations from uniform initialization and taking the resulting center frequencies.

### 4.3 PO-RSVMD Performance

Compared to standard RSVMD (from the paper, SNR 0-17 dB):
- Iteration time: reduced by at least **53%**
- Iteration count: reduced by at least **57%**
- RMSE: reduced by **35%**

### 4.4 Complete PO-RSVMD Pseudocode

```
COLD START: Same as RSVMD (Section 3.6)

SUBSEQUENT FRAMES (m = 1, 2, ...):
  1. Sliding DFT update (same as RSVMD)

  2. ADAPTIVE CENTER FREQUENCY INITIALIZATION:
     a. Compute omega_k^{detected}(m) via quick peak detection on current spectrum
     b. Compute delta_t = |iteration_time(m-1) - iteration_time(m-2)|
     c. Determine gamma from Formula 20 piecewise mapping
     d. omega_k^{init}(m) = gamma * omega_k^{final}(m-1) + (1-gamma) * omega_k^{detected}(m)

  3. ADMM LOOP WITH ERROR MUTATION CHECK:
     prev_error = infinity
     For n = 0 to max_iter:
       // Standard ADMM updates (mode, center freq, dual variable)

       curr_error = || sum_k hat{u}_k^{n+1} - hat{f} ||_2^2

       if curr_error > prev_error:    // Error mutation!
         Roll back to iteration n
         break

       if converged(tol_abs, tol_rel):
         break

       prev_error = curr_error

  4. Record iteration_time for next frame's gamma adaptation
  5. Boundary handling + output (same as RSVMD)
```

---

## 5. Rust Architecture

### 5.1 Crate Structure

```
rsvmd/
  Cargo.toml
  src/
    lib.rs              # PyO3 module registration
    vmd_core.rs         # Standard VMD ADMM solver (frequency domain)
    sliding_dft.rs      # Recursive SDFT implementation
    scale_space.rs      # Scale-space peak picking
    rsvmd.rs            # RSVMD processor (sliding + warm-start + VMD)
    po_rsvmd.rs         # PO-RSVMD processor (+ error mutation + adaptive gamma)
    complex_utils.rs    # Complex arithmetic helpers
    python.rs           # PyO3 wrapper classes
  benches/
    benchmarks.rs       # Criterion benchmarks
```

### 5.2 Core Types

```rust
use num_complex::Complex64;

/// Per-frame VMD state, persisted between frames
pub struct VmdState {
    /// DFT of current window: hat{f}(omega), length N
    signal_spectrum: Vec<Complex64>,

    /// Mode spectra: hat{u}_k(omega), shape K x N
    mode_spectra: Vec<Vec<Complex64>>,

    /// Center frequencies: omega_k, length K
    center_freqs: Vec<f64>,

    /// Lagrangian multiplier: hat{lambda}(omega), length N
    lambda: Vec<Complex64>,

    /// Whether cold start has been performed
    initialized: bool,

    /// Samples in current window (for boundary handling)
    window_buffer: Vec<f64>,
}

/// Configuration parameters
pub struct VmdConfig {
    pub alpha: f64,          // bandwidth constraint (e.g., 2000.0)
    pub k: usize,            // number of modes
    pub tau: f64,             // dual ascent step (0.0 for noiseless)
    pub tol: f64,             // convergence tolerance (1e-7)
    pub window_len: usize,   // N
    pub step_size: usize,     // s (sliding step)
    pub max_iter: usize,      // max ADMM iterations per frame
    pub damping: f64,         // SDFT damping factor (0.99999)
    pub fft_reset_interval: usize,  // recompute full FFT every P frames (0 = never)
}

/// PO-RSVMD specific config
pub struct PoRsvmdConfig {
    pub base: VmdConfig,
    pub gamma_default: f64,  // default rate factor when delta_t < 0.1 (0.5)
    /// Piecewise mapping from delta_t thresholds to gamma values (Formula 20)
    /// Entries: [(threshold, gamma_value), ...] sorted descending by threshold
    /// Default: [(0.8, 0.0), (0.6, 0.001), (0.4, 0.01), (0.2, 0.05), (0.1, 0.2)]
    pub gamma_tiers: Vec<(f64, f64)>,
}
```

### 5.3 Sliding DFT Module

```rust
/// Recursive sliding DFT processor
pub struct SlidingDft {
    /// Current DFT bins, length N
    bins: Vec<Complex64>,

    /// Precomputed twiddle factors: W_N^k for k = 0..N
    twiddles: Vec<Complex64>,

    /// Window length
    n: usize,

    /// Damping factor for numerical stability
    damping: f64,

    /// Precomputed r^{-N} for damped variant
    damping_inv_n: f64,

    /// Frame counter for periodic FFT reset
    frame_count: usize,

    /// How often to reset with full FFT (0 = never)
    fft_reset_interval: usize,
}

impl SlidingDft {
    /// Initialize with full FFT of first window
    pub fn new(initial_window: &[f64], damping: f64, fft_reset_interval: usize) -> Self;

    /// Slide by one sample: O(N) update
    /// x_new = sample entering, x_old = sample leaving
    pub fn slide_one(&mut self, x_new: f64, x_old: f64);

    /// Slide by s samples: O(N*s) update via UVT
    pub fn slide_block(&mut self, new_samples: &[f64], old_samples: &[f64]);

    /// Force full FFT recomputation from window buffer (reset drift)
    pub fn reset_from_buffer(&mut self, window: &[f64]);

    /// Get current spectrum (read-only)
    pub fn spectrum(&self) -> &[Complex64];
}
```

**Slide-one implementation detail:**

```rust
fn slide_one(&mut self, x_new: f64, x_old: f64) {
    let diff = Complex64::new(x_new - self.damping_inv_n * x_old, 0.0);
    for k in 0..self.n {
        self.bins[k] = self.damping * self.twiddles[k] * (self.bins[k] + diff);
        // Note: for block processing, diff varies per k (UVT)
    }
    self.frame_count += 1;
    if self.fft_reset_interval > 0 && self.frame_count % self.fft_reset_interval == 0 {
        // Periodic reset to prevent drift
        // Caller must provide current window buffer
    }
}
```

**Block slide with UVT:**

```rust
fn slide_block(&mut self, new_samples: &[f64], old_samples: &[f64]) {
    let s = new_samples.len();
    assert_eq!(s, old_samples.len());

    // Compute UVT: D_s[k] = sum_{j=0}^{s-1} (x_new[j] - x_old[j]) * W_N^{-kj}
    for k in 0..self.n {
        let mut d = Complex64::new(0.0, 0.0);
        for j in 0..s {
            let diff = new_samples[j] - old_samples[j];
            d += Complex64::new(diff, 0.0) * self.twiddles[k].conj().powi(j as i32);
        }
        // X_{m+s}[k] = W_N^{ks} * (X_m[k] + D_s[k])
        self.bins[k] = self.twiddles[k].powi(s as i32) * (self.bins[k] + d);
    }
}
```

### 5.4 VMD Core Solver

```rust
/// Single-frame VMD ADMM solver operating on a pre-computed spectrum
pub struct VmdSolver {
    config: VmdConfig,
    /// Precomputed normalized frequency array [0, 1/N, 2/N, ..., (N-1)/N]
    freqs: Vec<f64>,
}

impl VmdSolver {
    /// Run ADMM iterations on the given spectrum
    /// Returns (mode_spectra, center_freqs, iterations_used)
    pub fn solve(
        &self,
        signal_spectrum: &[Complex64],
        init_modes: Option<&[Vec<Complex64>]>,   // warm start
        init_center_freqs: Option<&[f64]>,         // warm start
        init_lambda: Option<&[Complex64]>,         // warm start
    ) -> VmdResult;

    /// Single ADMM iteration step (for fine-grained control)
    pub fn admm_step(
        &self,
        signal_spectrum: &[Complex64],
        mode_spectra: &mut [Vec<Complex64>],
        center_freqs: &mut [f64],
        lambda: &mut [Complex64],
    ) -> f64;  // returns convergence metric
}

pub struct VmdResult {
    pub mode_spectra: Vec<Vec<Complex64>>,  // K x N
    pub center_freqs: Vec<f64>,              // K
    pub lambda: Vec<Complex64>,              // N
    pub iterations: usize,
    pub converged: bool,
    pub final_error: f64,
}
```

**Mode update (Wiener filter) — the hot inner loop:**

```rust
fn update_mode(
    signal_spectrum: &[Complex64],
    other_modes_sum: &[Complex64],  // precomputed sum of all other modes
    lambda: &[Complex64],
    alpha: f64,
    omega_k: f64,
    freqs: &[f64],
    mode_out: &mut [Complex64],
) {
    let n = signal_spectrum.len();
    for i in 0..n {
        let numerator = signal_spectrum[i] - other_modes_sum[i] + lambda[i] * 0.5;
        let freq_diff = freqs[i] - omega_k;
        let denominator = 1.0 + 2.0 * alpha * freq_diff * freq_diff;
        mode_out[i] = numerator / denominator;
    }
}
```

**Center frequency update:**

```rust
fn update_center_freq(mode_spectrum: &[Complex64], freqs: &[f64], n: usize) -> f64 {
    let half_n = n / 2;
    let mut weighted_sum = 0.0;
    let mut power_sum = 0.0;

    // Only positive frequencies (0 to N/2)
    for i in 0..=half_n {
        let power = mode_spectrum[i].norm_sqr();  // |hat{u}_k|^2
        weighted_sum += freqs[i] * power;
        power_sum += power;
    }

    if power_sum > 1e-30 {
        weighted_sum / power_sum
    } else {
        0.0
    }
}
```

### 5.5 RSVMD Processor

```rust
pub struct RsvmdProcessor {
    config: VmdConfig,
    solver: VmdSolver,
    sdft: Option<SlidingDft>,
    state: VmdState,
    window_buffer: VecDeque<f64>,  // circular buffer of current window
}

impl RsvmdProcessor {
    pub fn new(config: VmdConfig) -> Self;

    /// Cold start: accepts exactly window_len samples
    /// Computes full FFT, runs scale-space peak picking, runs VMD to convergence
    pub fn initialize(&mut self, window: &[f64]) -> RsvmdOutput;

    /// Warm update: accepts exactly step_size samples
    /// Uses sliding DFT, warm-starts ADMM from previous state
    pub fn update(&mut self, new_samples: &[f64]) -> RsvmdOutput;

    /// Get current center frequencies
    pub fn center_freqs(&self) -> &[f64];

    /// Force FFT reset (call periodically if needed)
    pub fn reset_fft(&mut self);
}

pub struct RsvmdOutput {
    /// Decomposed modes in time domain, shape K x window_len
    pub modes: Vec<Vec<f64>>,
    /// Center frequencies, length K
    pub center_freqs: Vec<f64>,
    /// Number of ADMM iterations used
    pub iterations: usize,
    /// Whether ADMM converged
    pub converged: bool,
}
```

### 5.6 PO-RSVMD Processor

```rust
pub struct PoRsvmdProcessor {
    inner: RsvmdProcessor,
    po_config: PoRsvmdConfig,

    /// Iteration time history for gamma adaptation
    prev_iteration_time: Option<Duration>,

    /// Current adaptive gamma
    gamma: f64,
}

impl PoRsvmdProcessor {
    pub fn new(config: PoRsvmdConfig) -> Self;

    pub fn initialize(&mut self, window: &[f64]) -> RsvmdOutput;

    /// Update with error mutation detection and adaptive gamma
    pub fn update(&mut self, new_samples: &[f64]) -> RsvmdOutput;
}
```

**Error mutation detection in the ADMM loop:**

```rust
fn solve_with_mutation_check(
    &self,
    signal_spectrum: &[Complex64],
    state: &mut VmdState,
) -> VmdResult {
    let mut prev_error = f64::INFINITY;
    let mut best_modes = state.mode_spectra.clone();
    let mut best_freqs = state.center_freqs.clone();

    for n in 0..self.config.max_iter {
        // Standard ADMM step
        let curr_error = self.solver.admm_step(
            signal_spectrum,
            &mut state.mode_spectra,
            &mut state.center_freqs,
            &mut state.lambda,
        );

        // Error mutation check
        if curr_error > prev_error {
            // Over-decomposition detected — roll back
            state.mode_spectra = best_modes;
            state.center_freqs = best_freqs;
            return VmdResult {
                iterations: n,
                converged: false,  // stopped by mutation, not convergence
                final_error: prev_error,
                ..
            };
        }

        best_modes = state.mode_spectra.clone();
        best_freqs = state.center_freqs.clone();
        prev_error = curr_error;

        // Standard convergence check
        if self.check_convergence(&state.mode_spectra, &best_modes) {
            return VmdResult { iterations: n, converged: true, .. };
        }
    }

    VmdResult { iterations: self.config.max_iter, converged: false, .. }
}
```

**Adaptive gamma computation:**

```rust
fn compute_gamma(&mut self, iteration_time: Duration) -> f64 {
    if let Some(prev_time) = self.prev_iteration_time {
        let delta_t = (iteration_time.as_secs_f64() - prev_time.as_secs_f64()).abs();

        // Formula 20: piecewise mapping from delta_t to rate factor
        for &(threshold, gamma_val) in &self.po_config.gamma_tiers {
            if delta_t >= threshold {
                return gamma_val;
            }
        }
        self.po_config.gamma_default  // delta_t < smallest threshold → stable
    } else {
        self.po_config.gamma_default
    }
}

fn blend_center_freqs(
    prev_freqs: &[f64],
    detected_freqs: &[f64],
    gamma: f64,
) -> Vec<f64> {
    prev_freqs.iter().zip(detected_freqs.iter())
        .map(|(&prev, &det)| gamma * prev + (1.0 - gamma) * det)
        .collect()
}
```

### 5.7 Scale-Space Peak Picking

```rust
pub struct ScaleSpacePeakPicker {
    /// Number of smoothing scales
    n_scales: usize,
    /// Minimum sigma (Gaussian width)
    sigma_min: f64,
    /// Maximum sigma
    sigma_max: f64,
}

impl ScaleSpacePeakPicker {
    /// Find K most persistent peaks in the power spectrum
    pub fn pick_peaks(&self, power_spectrum: &[f64], k: usize) -> Vec<f64>;
}
```

**Algorithm:**

```rust
fn pick_peaks(&self, power_spectrum: &[f64], k: usize) -> Vec<f64> {
    let n = power_spectrum.len();
    let mut scores = vec![0.0f64; n];
    let mut smoothed = power_spectrum.to_vec();

    let sigmas = linspace(self.sigma_min, self.sigma_max, self.n_scales);

    for sigma in sigmas {
        // Apply Gaussian smoothing
        gaussian_smooth_inplace(&mut smoothed, sigma);

        // Find local maxima
        for i in 1..n-1 {
            if smoothed[i] > smoothed[i-1] && smoothed[i] > smoothed[i+1] {
                scores[i] += sigma;  // persistence score weighted by scale
            }
        }
    }

    // Return top-K peaks by persistence score
    let mut peak_indices: Vec<usize> = (0..n).collect();
    peak_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

    peak_indices.iter()
        .take(k)
        .map(|&i| i as f64 / n as f64)  // convert to normalized frequency
        .collect()
}
```

---

## 6. Python Bindings

### 6.1 PyO3 Module

```rust
// src/python.rs
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};

#[pyclass]
pub struct RSVMDProcessor {
    inner: crate::rsvmd::RsvmdProcessor,
}

#[pymethods]
impl RSVMDProcessor {
    #[new]
    #[pyo3(signature = (alpha=2000.0, k=3, tau=0.0, tol=1e-7, window_len=7200, step_size=1, max_iter=500, damping=0.99999, fft_reset_interval=0))]
    fn new(
        alpha: f64, k: usize, tau: f64, tol: f64,
        window_len: usize, step_size: usize, max_iter: usize,
        damping: f64, fft_reset_interval: usize,
    ) -> Self;

    /// Process samples. First call = cold start (len must == window_len).
    /// Subsequent calls = warm update (len must == step_size).
    /// Returns (modes: ndarray[K, window_len], center_freqs: ndarray[K])
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        samples: PyReadonlyArray1<f64>,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)>;

    /// Get current center frequencies
    fn center_freqs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    /// Force FFT reset
    fn reset_fft(&mut self);

    /// Check if processor has been initialized (cold start done)
    #[getter]
    fn initialized(&self) -> bool;
}

#[pyclass]
pub struct PORSVMDProcessor {
    inner: crate::po_rsvmd::PoRsvmdProcessor,
}

#[pymethods]
impl PORSVMDProcessor {
    #[new]
    #[pyo3(signature = (alpha=2000.0, k=3, tau=0.0, tol=1e-7, window_len=7200, step_size=1, max_iter=500, damping=0.99999, fft_reset_interval=0, gamma_default=0.5, gamma_tiers=None))]
    /// gamma_tiers: Optional list of (threshold, gamma_value) tuples sorted descending.
    /// Defaults to Formula 20: [(0.8, 0.0), (0.6, 0.001), (0.4, 0.01), (0.2, 0.05), (0.1, 0.2)]
    fn new(/* ... */) -> Self;

    fn update<'py>(
        &mut self,
        py: Python<'py>,
        samples: PyReadonlyArray1<f64>,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)>;

    fn center_freqs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64>;

    fn reset_fft(&mut self);

    /// Get the current adaptive gamma value
    #[getter]
    fn gamma(&self) -> f64;

    #[getter]
    fn initialized(&self) -> bool;
}

#[pymodule]
fn rsvmd(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RSVMDProcessor>()?;
    m.add_class::<PORSVMDProcessor>()?;
    Ok(())
}
```

### 6.2 Numpy Interop

All inputs/outputs use `numpy.ndarray[float64]` via the `numpy` crate for PyO3:
- Input samples: `PyReadonlyArray1<f64>` (zero-copy from numpy)
- Output modes: `PyArray2<f64>` of shape `(K, window_len)`
- Output center_freqs: `PyArray1<f64>` of shape `(K,)`

---

## 7. Project Scaffolding

### 7.1 Repository Structure

```
rsvmd-py/
  CLAUDE.md              # This file (or reference to it)
  LICENSE                 # MIT
  README.md
  Cargo.toml
  pyproject.toml          # maturin build config
  src/
    lib.rs
    vmd_core.rs
    sliding_dft.rs
    scale_space.rs
    rsvmd.rs
    po_rsvmd.rs
    complex_utils.rs
    python.rs
  tests/
    test_rsvmd.py         # Python integration tests
    test_po_rsvmd.py
    test_sliding_dft.py
    test_scale_space.py
  benches/
    benchmarks.rs         # Rust benchmarks (criterion)
  examples/
    basic_usage.py
    streaming.py
  .github/
    workflows/
      ci.yml              # Build + test on Linux/macOS/Windows
      release.yml         # Build wheels + publish to PyPI
```

### 7.2 Cargo.toml

```toml
[package]
name = "rsvmd"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Recursive Sliding Variational Mode Decomposition with Python bindings"

[lib]
name = "rsvmd"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
num-complex = "0.4"
rustfft = "6.2"         # For cold-start full FFT and periodic resets
rayon = "1.10"           # Optional: parallelize across frequency bins

[dev-dependencies]
criterion = "0.5"
approx = "0.5"           # Float comparison in tests

[[bench]]
name = "benchmarks"
harness = false
```

### 7.3 pyproject.toml

```toml
[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[project]
name = "rsvmd"
version = "0.1.0"
description = "Recursive Sliding Variational Mode Decomposition (RSVMD & PO-RSVMD)"
requires-python = ">=3.9"
license = "MIT"
dependencies = ["numpy>=1.20"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "scipy>=1.10"]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "rsvmd"
```

### 7.4 CI Workflow (.github/workflows/ci.yml)

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - run: pip install maturin[patchelf] pytest numpy scipy
      - run: maturin develop
      - run: cargo test
      - run: pytest tests/
```

---

## 8. Test Plan

### 8.1 Rust Unit Tests

**sliding_dft.rs:**
- Compute full FFT of a known signal (e.g., sum of 3 sinusoids). Slide one sample. Compute full FFT of shifted window. Assert SDFT result matches within 1e-10.
- Slide 1000 samples one at a time, compare final result to full FFT. Measure accumulated drift.
- Test damping factor: slide 10000 samples, verify drift stays bounded.
- Block slide: slide by s=10 samples, compare to s individual single slides.

**vmd_core.rs:**
- Decompose signal = sin(2*pi*f1*t) + sin(2*pi*f2*t) + sin(2*pi*f3*t) with f1=1, f2=5, f3=20 Hz, K=3. Verify center frequencies converge near 1, 5, 20.
- Verify reconstruction: sum of modes approximates original signal within tol.
- Test convergence: verify iteration count is within expected range.
- Test warm start vs cold start: warm start should converge in fewer iterations.

**scale_space.rs:**
- Create spectrum with 3 clear peaks and noise. Verify pick_peaks returns 3 frequencies near the true peaks.
- Create spectrum with noise only. Verify it doesn't crash and returns some frequencies.

**rsvmd.rs:**
- Process a 30-second signal frame by frame. Compare modes to batch VMD result. They should be similar (not identical due to windowing, but center frequencies should match).
- Verify warm-start iteration count is < 10 for frames after the first.

**po_rsvmd.rs:**
- Test error mutation: create a signal where standard VMD over-decomposes (error increases after N iterations). Verify PO-RSVMD stops early.
- Test gamma adaptation: simulate a signal that changes character mid-stream. Verify gamma decreases (more weight on fresh detection).

### 8.2 Python Integration Tests

**test_rsvmd.py:**
```python
def test_basic_decomposition():
    """3 sinusoids → 3 modes with correct center frequencies."""

def test_streaming_consistency():
    """Process same signal batch vs streaming, verify similar results."""

def test_warm_start_faster():
    """Second frame converges in fewer iterations than first."""

def test_numpy_shapes():
    """Output arrays have correct shapes (K, window_len) and (K,)."""
```

**test_po_rsvmd.py:**
```python
def test_over_decomposition_prevention():
    """Signal that causes standard RSVMD to over-decompose is handled."""

def test_gamma_adaptation():
    """Gamma adjusts when signal characteristics change."""

def test_matches_standard_on_clean_signal():
    """On clean signals, PO-RSVMD matches RSVMD quality."""
```

### 8.3 Benchmarks

**benchmarks.rs (criterion):**
- `bench_cold_start`: Full VMD on N=7200 signal, K=3. Baseline.
- `bench_sdft_slide_one`: Single SDFT update. Target: < 10 us for N=7200.
- `bench_warm_frame`: One warm-started ADMM solve. Target: < 1 ms for N=7200, K=3.
- `bench_e2e_1000_frames`: Process 1000 frames. Measure total throughput.

---

## 9. Integration Notes (ctrade)

### 9.1 Dependency

In ctrade's `pyproject.toml`:
```toml
dependencies = [
    "rsvmd>=0.1.0",
    # ... other deps
]
```

Remove `vmdpy` from dependencies.

### 9.2 Wrapper Update

Replace `src/ctrade/lib/vmd.py`:

```python
import numpy as np
from rsvmd import RSVMDProcessor, PORSVMDProcessor


class VmdDecomposer:
    """Wraps RSVMD for use in ctrade's Kedro pipeline.

    Handles both batch (training) and streaming (inference) modes.
    """

    def __init__(
        self,
        num_modes: int = 3,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        window_minutes: int = 30,
        bar_resolution_ms: int = 250,
        use_po: bool = True,
    ):
        self.window_len = int(window_minutes * 60 * 1000 / bar_resolution_ms)
        self.num_modes = num_modes
        self.use_po = use_po

        ProcessorClass = PORSVMDProcessor if use_po else RSVMDProcessor
        self._proc = ProcessorClass(
            alpha=alpha,
            k=num_modes,
            tau=tau,
            tol=tol,
            window_len=self.window_len,
            step_size=1,
            max_iter=500,
        )

    def decompose_batch(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Training mode: decompose full signal using sliding window.

        Returns:
            imfs: shape (K, len(signal))
            center_freqs: shape (K,) from final window
        """
        if len(signal) < self.window_len:
            # Pad or use full signal as single window
            padded = np.pad(signal, (0, self.window_len - len(signal)))
            imfs, cfreqs = self._proc.update(padded)
            return imfs[:, :len(signal)], cfreqs

        # Cold start with first window
        imfs, cfreqs = self._proc.update(signal[:self.window_len])

        # Slide through remaining samples
        all_imfs = [imfs]
        for i in range(self.window_len, len(signal)):
            imfs, cfreqs = self._proc.update(signal[i:i+1])
            # Only need the latest sample from each mode

        return imfs, cfreqs

    def update_online(self, new_sample: float) -> tuple[np.ndarray, np.ndarray]:
        """Inference mode: process one new sample.

        Returns:
            imfs: shape (K, window_len) - current window decomposition
            center_freqs: shape (K,)
        """
        return self._proc.update(np.array([new_sample]))
```

### 9.3 Node Function Update

In `src/ctrade/nodes/decomposition.py`, replace the `apply_vmd_decomposition` function to use `VmdDecomposer` instead of directly calling `vmdpy.VMD`. The key change: instead of independently decomposing overlapping windows, use RSVMD's streaming mode which naturally handles the sliding window with warm-starting.

### 9.4 Latency Impact

Current system budget: 230-500ms total pipeline latency.

With RSVMD:
- Cold start (first frame): ~50-100ms for N=7200, K=3 (one-time cost at startup)
- Warm frame: ~0.5-2ms per frame (2-5 ADMM iterations on pre-computed SDFT)
- vs. standard VMD per window: ~20-50ms (full FFT + 50-500 iterations)

This frees up significant latency budget for the MASS motif queries downstream.

---

## 10. References

### Papers

1. **VMD (foundational):** K. Dragomiretskiy and D. Zosso, "Variational Mode Decomposition," IEEE Trans. Signal Processing, vol. 62, no. 3, pp. 531-544, 2014. [UCLA Technical Report](https://ww3.math.ucla.edu/camreport/cam13-22.pdf)

2. **RSVMD (original):** "Block-wise recursive sliding variational mode decomposition method and its application on online separating of bridge vehicle-induced strain monitoring signals," Mechanical Systems and Signal Processing, May 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327023002960)

3. **PO-RSVMD:** "The Parameter-Optimized Recursive Sliding Variational Mode Decomposition Algorithm and Its Application in Sensor Signal Processing," Sensors, vol. 25, no. 6, article 1944, March 2025. [MDPI (open access)](https://www.mdpi.com/1424-8220/25/6/1944) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946174/)

4. **Recursive Windowed VMD:** "Recursive Windowed Variational Mode Decomposition," Circuits, Systems, and Signal Processing, 2024. [Springer](https://link.springer.com/article/10.1007/s00034-024-02864-2)

5. **Scale-Space Peak Picking:** [INRIA Technical Report](https://inria.hal.science/hal-01103123v2/document)

### Existing Implementations (reference only)

- [vmdpy (Python, archived)](https://github.com/vrcarva/vmdpy) — reference for standard VMD ADMM loop
- [vmdrs-py (Rust+Python)](https://github.com/jiafuei/vmdrs-py) — reference for PyO3 VMD bindings
- [sdft (Rust)](https://github.com/jurihock/sdft) — reference for Sliding DFT in Rust

### Important Note

The original RSVMD paper (reference 2) is behind a ScienceDirect paywall. The PO-RSVMD paper (reference 3) is open access and contains the standard VMD equations plus the PO-RSVMD extensions. **You should read the PO-RSVMD paper (reference 3) for the exact Formulas 18-20** which define the rate learning factor mapping. The adaptive gamma logic in this spec is a reasonable approximation but the paper's empirical mapping may differ.
