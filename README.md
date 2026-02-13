# rsvmd

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0--1.0-lightgrey.svg)](LICENSE)

Recursive Sliding Variational Mode Decomposition (RSVMD & PO-RSVMD) — a Rust implementation with Python bindings for real-time signal decomposition in streaming/sliding-window applications.

## Overview

**Variational Mode Decomposition (VMD)** decomposes a signal into a set of band-limited intrinsic mode functions (IMFs) by solving a constrained variational optimization problem. Standard VMD operates in batch mode, requiring a full FFT on each window — too slow for real-time pipelines.

**RSVMD** replaces the batch FFT with a recursive Sliding DFT, warm-starts each frame's ADMM solver from the previous frame's solution, and uses scale-space peak picking for robust cold-start initialization. **PO-RSVMD** extends this with error mutation detection (preventing over-decomposition) and adaptive center frequency blending (handling signal drift under interference).

### Key features

- **Streaming decomposition**: O(N) per frame via Sliding DFT, vs O(N log N) for batch FFT
- **Warm-starting**: ADMM converges in 2–10 iterations after cold start (vs 50–500)
- **PO-RSVMD**: Error mutation detection + adaptive gamma for robustness under interference
- **Zero-copy Python interop**: numpy arrays in/out via PyO3, no data copies
- **Result-based error handling**: no panics in the Rust core; all errors returned as `PyErr`
- **113 tests**: 61 Rust unit tests + 52 Python integration tests, including verification of claims from [1,2,3]

## Installation

### Prerequisites

- Python 3.13+
- Rust toolchain (install via [rustup](https://rustup.rs/))
- [maturin](https://www.maturin.rs/) (`uv pip install maturin`)

### Build from source

```bash
# Build and install into current venv
maturin develop --uv

# For benchmarking / performance testing, use release mode
maturin develop --uv --release

# Install dev dependencies
uv pip install pytest scipy
```

## Quick Start

### RSVMD — streaming decomposition

```python
import numpy as np
from rsvmd import RSVMDProcessor

# Create a test signal: 3 sinusoids
N = 7200
t = np.arange(N + 100, dtype=np.float64) / N
signal = (np.sin(2 * np.pi * 10 * t)
        + 0.7 * np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 200 * t))

proc = RSVMDProcessor(alpha=2000.0, k=3, window_len=N)

# Cold start: pass exactly window_len samples
modes, center_freqs = proc.update(signal[:N])
# modes.shape == (3, 7200), center_freqs.shape == (3,)

# Streaming: pass exactly step_size samples (default 1)
for i in range(100):
    modes, center_freqs = proc.update(signal[N + i : N + i + 1])

print(f"Center frequencies: {center_freqs}")
print(f"Converged: {proc.last_converged}, Iterations: {proc.last_iterations}")
```

### PO-RSVMD — with over-decomposition prevention

```python
from rsvmd import PORSVMDProcessor

po_proc = PORSVMDProcessor(alpha=2000.0, k=3, window_len=7200)

modes, center_freqs = po_proc.update(signal[:N])

for i in range(100):
    modes, center_freqs = po_proc.update(signal[N + i : N + i + 1])

print(f"Adaptive gamma: {po_proc.gamma}")
```

### Output format

- **`modes`**: `ndarray[float64]` of shape `(K, window_len)` — the K decomposed IMFs in the time domain
- **`center_freqs`**: `ndarray[float64]` of shape `(K,)` — normalized center frequencies in [0, 0.5], where 0.5 corresponds to the Nyquist frequency

## API Reference

### RSVMDProcessor

```python
RSVMDProcessor(
    alpha=2000.0,          # bandwidth constraint penalty
    k=3,                   # number of modes to extract
    tau=0.0,               # noise tolerance (0 = exact reconstruction)
    tol=1e-7,              # ADMM convergence tolerance
    window_len=7200,       # window length in samples
    step_size=1,           # sliding step size
    max_iter=500,          # max ADMM iterations per frame
    damping=0.99999,       # SDFT damping factor for numerical stability
    fft_reset_interval=0,  # recompute full FFT every N frames (0 = never)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `2000.0` | Bandwidth constraint. Larger values produce narrower modes. |
| `k` | `int` | `3` | Number of modes to decompose into. |
| `tau` | `float` | `0.0` | Dual ascent step size. 0 = noiseless (exact reconstruction constraint). |
| `tol` | `float` | `1e-7` | ADMM convergence tolerance (relative change in modes). |
| `window_len` | `int` | `7200` | Number of samples in the sliding window. |
| `step_size` | `int` | `1` | Samples to slide per frame. First call must pass `window_len` samples; subsequent calls must pass `step_size` samples. |
| `max_iter` | `int` | `500` | Maximum ADMM iterations per frame (safety bound). |
| `damping` | `float` | `0.99999` | Sliding DFT damping factor. Values < 1 prevent numerical drift accumulation. |
| `fft_reset_interval` | `int` | `0` | Recompute a full FFT every N frames to reset drift. 0 disables. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `update(samples)` | `(ndarray, ndarray)` | Process samples. First call = cold start (`len == window_len`), subsequent = warm update (`len == step_size`). Returns `(modes, center_freqs)`. |
| `center_freqs()` | `ndarray` | Current center frequencies (shape `(K,)`). |
| `reset_fft()` | `None` | Force a full FFT recomputation from the current window buffer. |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `initialized` | `bool` | Whether cold start has been performed. |
| `last_iterations` | `int` | ADMM iterations used in the most recent `update()` call. |
| `last_converged` | `bool` | Whether the most recent `update()` converged within `tol`. |

### PORSVMDProcessor

Extends `RSVMDProcessor` with error mutation detection and adaptive center frequency initialization.

```python
PORSVMDProcessor(
    # All RSVMDProcessor parameters, plus:
    gamma_default=0.5,     # default rate learning factor
    gamma_tiers=None,      # custom piecewise mapping (see below)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma_default` | `float` | `0.5` | Rate learning factor when iteration time is stable (`delta_t < 0.1`). |
| `gamma_tiers` | `list[tuple[float, float]] \| None` | `None` | Piecewise `(threshold, gamma)` mapping sorted descending. Default: `[(0.8, 0.0), (0.6, 0.001), (0.4, 0.01), (0.2, 0.05), (0.1, 0.2)]` (Formula 20 from [3]). |

**Additional property:**

| Property | Type | Description |
|----------|------|-------------|
| `gamma` | `float` | Current adaptive gamma value. |

### Parameter guidance

- **`alpha`**: Controls mode bandwidth. Higher = narrower, more selective modes. Start with 2000 for well-separated spectral components; reduce for broadband signals.
- **`k`**: Set to the expected number of spectral components. Too high can cause over-decomposition (PO-RSVMD mitigates this). Too low merges components.
- **`tau`**: Keep at 0 for noiseless signals. Increase (e.g., 0.1–1.0) for noisy signals to allow reconstruction slack.
- **`window_len`**: Must capture at least a few periods of the lowest-frequency component. For 250ms bars with 30-minute windows: 7200 samples.

## Algorithm Background

### Standard VMD

VMD decomposes a signal f(t) into K modes {u_k} with center frequencies {omega_k} by minimizing aggregate bandwidth subject to a reconstruction constraint. The augmented Lagrangian is solved via ADMM with three alternating updates per iteration, all performed pointwise in the frequency domain:

**Mode update** (Wiener filter):

```
û_k(ω) = [f̂(ω) - Σ_{i≠k} û_i(ω) + λ̂(ω)/2] / [1 + 2α(ω - ω_k)²]
```

**Center frequency update** (power-weighted mean over positive frequencies):

```
ω_k = Σ_{ω≥0} ω|û_k(ω)|² / Σ_{ω≥0} |û_k(ω)|²
```

**Dual variable update**:

```
λ̂(ω) ← λ̂(ω) + τ(f̂(ω) - Σ_k û_k(ω))
```

Convergence is checked via relative change in mode spectra. Complexity per iteration: O(KN) after the initial O(N log N) FFT.

### RSVMD extensions

**Sliding DFT**: Replaces the batch FFT with a recursive update. When the window slides by one sample:

```
X_{m+1}[k] = r · W_N^k · (X_m[k] + x_new - r^{-N} · x_old)
```

where r is a damping factor (default 0.99999) for numerical stability. Cost: O(N) per slide.

**Warm-starting**: Each frame initializes modes and center frequencies from the previous frame's converged result. This reduces ADMM iterations from 50–500 (cold) to typically 2–10 (warm).

**Scale-space peak picking**: On cold start, the power spectrum is smoothed at multiple Gaussian kernel widths. Peaks that persist across scales receive high persistence scores. The top-K persistent peaks become initial center frequencies — more robust than simple peak detection.

### PO-RSVMD extensions

**Error mutation detection** (Section 4.1 of [3]): Monitors reconstruction error between ADMM iterations. If error *increases* (over-decomposition), the solver rolls back to the previous iteration and terminates early.

**Adaptive center frequency blending** (Formula 20 from [3]): Instead of directly warm-starting from previous center frequencies, blends them with freshly detected peaks:

```
ω_k^init(m) = γ · ω_k^final(m-1) + (1-γ) · ω_k^detected(m)
```

The rate factor γ is adapted based on iteration-time stability via a 6-tier piecewise mapping. Stable iteration times → high γ (trust history). Unstable → low γ (trust fresh detection).

### Complexity

| Operation | Cost |
|-----------|------|
| Cold start (FFT + scale-space + ADMM) | O(N log N + KN · max_iter) |
| Sliding DFT update (per sample) | O(N) |
| Warm ADMM frame | O(KN · iters), iters typically 2–10 |

## Architecture & Design

### Rust module layout

```
src/
├── lib.rs            # PyO3 module registration
├── vmd_core.rs       # ADMM solver: Wiener filter, center freq, convergence
├── sliding_dft.rs    # Recursive SDFT with damping and periodic reset
├── scale_space.rs    # Gaussian scale-space peak picking
├── rsvmd_core.rs     # RSVMD processor (SDFT + warm-start + VMD)
├── po_rsvmd.rs       # PO-RSVMD (error mutation + adaptive gamma)
├── complex_utils.rs  # Complex arithmetic helpers
└── python.rs         # PyO3 wrapper classes and numpy interop
```

### Design decisions

- **Rust for the inner loop**: The hot path is complex Wiener filtering per frequency bin × K modes × ADMM iterations × sliding frames. Rust eliminates Python overhead and enables SIMD-friendly data layouts.
- **Ownership for state safety**: Recursive state (DFT bins, center frequencies, mode spectra) persists across calls. Rust's ownership model prevents aliasing bugs that are common in stateful numerical code.
- **Gauss-Seidel mode ordering**: Mode updates use the most recently computed values for modes i < k (already updated this iteration) rather than all values from the previous iteration (Jacobi). This matches [1] and improves convergence.
- **Result-based errors**: All public Rust functions return `Result<_, _>`. Invalid inputs (wrong array length, update before init) produce Python exceptions, not panics.
- **Damped SDFT with periodic reset**: The damping factor r < 1 bounds rounding-error growth. Optional periodic full-FFT recomputation (`fft_reset_interval`) provides a hard reset for very long streams.
- **Zero-copy numpy interop**: Input arrays are read via `PyReadonlyArray1` (no copy). Output arrays are allocated as numpy arrays directly.

## Paper Validation & Test Suite

The test suite verifies both implementation correctness and specific claims from [1,2,3].

**Test counts**: 61 Rust unit tests + 52 Python integration tests = **113 tests total**.

### ADMM equation verification (Rust)

These tests verify that the core ADMM update formulas match the equations in [1]:

| Test | What it verifies |
|------|-----------------|
| `vmd_core::test_wiener_filter_formula_k1` | Mode update (Wiener filter) numerator/denominator for K=1 |
| `vmd_core::test_wiener_filter_gauss_seidel_k2` | Gauss-Seidel ordering: mode 1 uses mode 0's *updated* values |
| `vmd_core::test_center_freq_power_weighted_mean_single_bin` | Center frequency = power-weighted mean (single bin) |
| `vmd_core::test_center_freq_power_weighted_mean_two_bins` | Center frequency = power-weighted mean (two bins) |
| `vmd_core::test_dual_variable_update_formula` | Dual variable update with τ scaling |
| `vmd_core::test_convergence_criterion_formula` | Convergence via relative squared L2 norm change |

### Signal separation quality

| Test | What it verifies |
|------|-----------------|
| `vmd_core::test_mode_spectral_concentration` | Each mode's energy is concentrated near its center frequency (>50%) |
| `vmd_core::test_mode_cross_correlation_low` | Modes are nearly orthogonal (cross-correlation < 0.3) |
| `vmd_core::test_reconstruction_error_decreases` | Reconstruction error monotonically decreases across iterations |
| `test_rsvmd::TestPaperClaims::test_mode_spectral_separation` | Python-side spectral concentration check |
| `test_rsvmd::TestPaperClaims::test_mode_cross_correlation_low` | Python-side cross-correlation check |

### RSVMD claims [2]

| Claim | Test |
|-------|------|
| Streaming RSVMD matches batch VMD center frequencies | `rsvmd_core::test_rsvmd_matches_batch_vmd_on_stationary_signal` |
| Warm frames converge in 2–5 iterations (with τ=0) | `rsvmd_core::test_warm_converges_within_5` |
| Warm frames converge in fewer iterations than cold start | `rsvmd_core::test_warm_start_fewer_iterations` |
| Center frequencies remain stable across streaming | `rsvmd_core::test_center_freq_stability_across_streaming` |
| Long streaming (200 frames) produces no NaN/Inf | `rsvmd_core::test_long_streaming_200_frames` |
| Python: warm converges within 10 iterations | `test_rsvmd::TestPaperClaims::test_warm_converges_within_10` |
| Python: RSVMD matches batch VMD | `test_rsvmd::TestPaperClaims::test_rsvmd_matches_batch_vmd` |

### PO-RSVMD claims [3]

| Claim | Test |
|-------|------|
| Error mutation detected and stops iteration early | `po_rsvmd::test_error_mutation_detection_stops_early` |
| Gamma adapts under signal change | `po_rsvmd::test_gamma_adaptation_under_signal_change` |
| Gamma blending formula matches boundary values | `po_rsvmd::test_gamma_blending_formula_boundary_values` |
| PO-RSVMD reduces iteration count vs standard RSVMD | `po_rsvmd::test_po_warm_start_fewer_iterations` |
| Error mutation principle: error can increase past optimum | `vmd_core::test_error_mutation_principle` |
| Python: error mutation prevents over-decomposition | `test_po_rsvmd::TestPOPaperClaims::test_po_error_mutation_prevents_overdecmp` |
| Python: gamma blending formula verification | `test_po_rsvmd::TestPOPaperClaims::test_po_gamma_blending_formula` |
| Python: PO-RSVMD reduces iteration count | `test_po_rsvmd::TestPOPaperClaims::test_po_reduces_iteration_count` |
| Python: spectral separation maintained | `test_po_rsvmd::TestPOPaperClaims::test_po_mode_spectral_separation` |
| Python: low cross-correlation between modes | `test_po_rsvmd::TestPOPaperClaims::test_po_cross_correlation_low` |

### Scale-space peak picking

| Test | What it verifies |
|------|-----------------|
| `scale_space::test_pick_peaks_three_sinusoids` | Detects 3 true peaks from a multi-sinusoid spectrum |
| `scale_space::test_peaks_robust_to_noise` | Peak detection works through additive noise |
| `scale_space::test_pick_peaks_fills_missing` | Returns K peaks even when fewer are prominent |
| `scale_space::test_gaussian_smooth` | Gaussian smoothing kernel is correct |

### Stability and edge cases

Additional tests cover: zero signals, constant signals, single-mode decomposition (K=1), K larger than signal content, step_size > 1, FFT reset intervals, 200-frame long streaming, error handling for wrong input sizes, and uninitialized processor access.

## Benchmarks & Performance

### Criterion benchmarks (Rust)

| Benchmark | Description |
|-----------|-------------|
| `bench_cold_start` | Full VMD cold start: FFT + scale-space + ADMM (N=7200, K=3) |
| `bench_sdft_slide_one` | Single sliding DFT update (N=7200) |
| `bench_warm_frame` | One warm-started ADMM frame (N=7200, K=3) |
| `bench_e2e_100_frames` | End-to-end: cold start + 100 warm frames |

```bash
cargo bench
```

### Python performance test

Asserts end-to-end timing from Python (release build):

| Metric | Limit |
|--------|-------|
| Cold start | < 500 ms |
| Warm frame median | < 50 ms |

```bash
maturin develop --uv --release
python tests/test_performance.py
```

## Project Structure

```
rsvmd-py/
├── Cargo.toml                  # Rust crate config
├── pyproject.toml              # Python package config (maturin)
├── CLAUDE.md                   # Full algorithm spec and design doc
├── README.md                   # This file
├── src/
│   ├── lib.rs                  # PyO3 module entry point
│   ├── vmd_core.rs             # ADMM solver (Wiener filter, center freq, dual update)
│   ├── sliding_dft.rs          # Recursive SDFT with damping
│   ├── scale_space.rs          # Scale-space peak picker
│   ├── rsvmd_core.rs           # RSVMD streaming processor
│   ├── po_rsvmd.rs             # PO-RSVMD with error mutation + adaptive gamma
│   ├── complex_utils.rs        # Complex number helpers
│   └── python.rs               # PyO3 bindings and numpy interop
├── python/
│   └── rsvmd/
│       ├── __init__.py          # Re-exports RSVMDProcessor, PORSVMDProcessor
│       └── __init__.pyi         # Type stubs for IDE support
├── tests/
│   ├── test_rsvmd.py            # 27 Python tests for RSVMDProcessor
│   ├── test_po_rsvmd.py         # 25 Python tests for PORSVMDProcessor
│   └── test_performance.py      # Manual performance benchmark
└── benches/
    └── benchmarks.rs            # Criterion benchmarks
```

## Development

### Build and test

```bash
# Build debug extension
maturin develop --uv

# Run Rust tests
cargo test

# Run Python tests
pytest tests/

# Run all
cargo test && pytest tests/
```

### Type checking

```bash
pyright python/
```

### Benchmarks

```bash
# Rust micro-benchmarks
cargo bench

# Python end-to-end performance (requires release build)
maturin develop --uv --release
python tests/test_performance.py
```

## References

### Papers

1. **VMD** (foundational): K. Dragomiretskiy and D. Zosso, "Variational Mode Decomposition," *IEEE Trans. Signal Processing*, vol. 62, no. 3, pp. 531–544, 2014. [UCLA Technical Report](https://ww3.math.ucla.edu/camreport/cam13-22.pdf)

2. **RSVMD**: "Block-wise recursive sliding variational mode decomposition method and its application on online separating of bridge vehicle-induced strain monitoring signals," *Mechanical Systems and Signal Processing*, May 2023. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0888327023002960)

3. **PO-RSVMD**: "The Parameter-Optimized Recursive Sliding Variational Mode Decomposition Algorithm and Its Application in Sensor Signal Processing," *Sensors*, vol. 25, no. 6, article 1944, March 2025. [MDPI (open access)](https://www.mdpi.com/1424-8220/25/6/1944) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946174/)

4. **Recursive Windowed VMD**: "Recursive Windowed Variational Mode Decomposition," *Circuits, Systems, and Signal Processing*, 2024. [Springer](https://link.springer.com/article/10.1007/s00034-024-02864-2)

5. **Scale-Space Peak Picking**: [INRIA Technical Report](https://inria.hal.science/hal-01103123v2/document)

### Reference implementations

- [vmdpy](https://github.com/vrcarva/vmdpy) — Python VMD (archived)
- [vmdrs-py](https://github.com/jiafuei/vmdrs-py) — Rust+Python VMD bindings
- [sdft](https://github.com/jurihock/sdft) — Sliding DFT in Rust

## License

CC0-1.0
