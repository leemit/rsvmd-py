import numpy as np
import pytest
from rsvmd import RSVMDProcessor


def make_signal(n, freqs=(10.0, 50.0, 200.0), amplitudes=(1.0, 0.7, 0.5)):
    """Generate a test signal as sum of sinusoids."""
    t = np.arange(n, dtype=np.float64) / n
    signal = np.zeros(n, dtype=np.float64)
    for f, a in zip(freqs, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
    return signal


class TestBasicDecomposition:
    def test_three_sinusoids(self):
        """3 sinusoids -> 3 modes with correct center frequencies."""
        n = 512
        signal = make_signal(n, freqs=(10, 50, 200))

        proc = RSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        modes, center_freqs = proc.update(signal)

        assert modes.shape == (3, n)
        assert center_freqs.shape == (3,)
        assert proc.initialized

        # Center frequencies should be near the true values (normalized)
        sorted_freqs = np.sort(center_freqs)
        expected = np.array([10.0 / n, 50.0 / n, 200.0 / n])
        tol = 15.0 / n
        np.testing.assert_allclose(sorted_freqs, expected, atol=tol)

    def test_two_sinusoids(self):
        """2 sinusoids -> 2 modes."""
        n = 256
        signal = make_signal(n, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )

        modes, center_freqs = proc.update(signal)
        assert modes.shape == (2, n)
        assert center_freqs.shape == (2,)


class TestStreamingConsistency:
    def test_streaming_produces_valid_output(self):
        """Process same signal batch vs streaming, verify shapes are correct."""
        n = 256
        n_steps = 10
        signal = make_signal(n + n_steps, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        # Cold start
        modes, cfreqs = proc.update(signal[:n])
        assert modes.shape == (2, n)

        # Streaming updates
        for i in range(n_steps):
            modes, cfreqs = proc.update(signal[n + i : n + i + 1])
            assert modes.shape == (2, n)
            assert cfreqs.shape == (2,)


class TestNumpyShapes:
    def test_output_shapes(self):
        """Output arrays have correct shapes (K, window_len) and (K,)."""
        n = 128
        k = 4
        signal = make_signal(n)

        proc = RSVMDProcessor(
            alpha=2000.0, k=k, tau=0.0, tol=1e-7,
            window_len=n, max_iter=500,
        )

        modes, center_freqs = proc.update(signal)
        assert modes.shape == (k, n)
        assert center_freqs.shape == (k,)
        assert modes.dtype == np.float64
        assert center_freqs.dtype == np.float64

    def test_center_freqs_method(self):
        """center_freqs() method returns correct array."""
        n = 128
        k = 3
        signal = make_signal(n)

        proc = RSVMDProcessor(alpha=2000.0, k=k, window_len=n, max_iter=500)
        proc.update(signal)

        cfreqs = proc.center_freqs()
        assert cfreqs.shape == (k,)
        assert cfreqs.dtype == np.float64


class TestErrorHandling:
    def test_wrong_initial_size(self):
        """First call with wrong number of samples raises error."""
        proc = RSVMDProcessor(window_len=256, k=3)
        with pytest.raises(ValueError, match="First call must provide"):
            proc.update(np.zeros(100))

    def test_wrong_update_size(self):
        """Subsequent call with wrong step_size raises error."""
        n = 128
        proc = RSVMDProcessor(window_len=n, step_size=1, k=2)
        proc.update(np.zeros(n))
        with pytest.raises(ValueError, match="Expected 1 samples"):
            proc.update(np.zeros(5))

    def test_not_initialized_property(self):
        """initialized property returns False before first call."""
        proc = RSVMDProcessor(window_len=128, k=3)
        assert not proc.initialized


class TestResetFft:
    def test_reset_doesnt_crash(self):
        """reset_fft() can be called after initialization."""
        n = 128
        signal = make_signal(n)
        proc = RSVMDProcessor(window_len=n, k=2)
        proc.update(signal)
        proc.reset_fft()
        # Should still produce valid output after reset
        modes, cfreqs = proc.update(np.zeros(1))
        assert modes.shape == (2, n)


class TestReconstructionQuality:
    def test_modes_sum_approximates_signal(self):
        """Sum of decomposed modes should approximate the original signal."""
        n = 512
        signal = make_signal(n, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        modes, _ = proc.update(signal)

        reconstructed = modes.sum(axis=0)
        error = np.linalg.norm(reconstructed - signal) / np.linalg.norm(signal)
        assert error < 1.0, f"Reconstruction error too high: {error:.4f}"

    def test_no_nan_in_output(self):
        """Output should never contain NaN values."""
        n = 256
        signal = make_signal(n)

        proc = RSVMDProcessor(alpha=2000.0, k=3, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(signal)

        assert not np.any(np.isnan(modes)), "NaN found in modes"
        assert not np.any(np.isnan(cfreqs)), "NaN found in center_freqs"


class TestEdgeCases:
    def test_single_mode_k1(self):
        """K=1 should produce a single valid mode."""
        n = 256
        signal = make_signal(n, freqs=(20,), amplitudes=(1.0,))

        proc = RSVMDProcessor(alpha=2000.0, k=1, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(signal)

        assert modes.shape == (1, n)
        assert cfreqs.shape == (1,)
        assert not np.any(np.isnan(modes))

    def test_large_k_more_modes_than_content(self):
        """K > number of spectral peaks should still work."""
        n = 256
        signal = make_signal(n, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(alpha=2000.0, k=5, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(signal)

        assert modes.shape == (5, n)
        assert cfreqs.shape == (5,)
        assert not np.any(np.isnan(modes))
        assert not np.any(np.isnan(cfreqs))

    def test_zero_signal(self):
        """Zero signal should not crash or produce NaN."""
        n = 128
        proc = RSVMDProcessor(alpha=2000.0, k=2, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(np.zeros(n))

        assert modes.shape == (2, n)
        assert not np.any(np.isnan(modes))
        assert not np.any(np.isnan(cfreqs))

    def test_constant_signal(self):
        """Constant signal should not crash or produce NaN."""
        n = 128
        proc = RSVMDProcessor(alpha=2000.0, k=2, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(np.full(n, 3.14))

        assert modes.shape == (2, n)
        assert not np.any(np.isnan(modes))
        assert not np.any(np.isnan(cfreqs))


class TestStepSizeGreaterThanOne:
    def test_step_size_5(self):
        """Streaming with step_size=5 produces valid output."""
        n = 256
        step = 5
        total = n + step * 5
        signal = make_signal(total, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=step, max_iter=500,
        )

        modes, _ = proc.update(signal[:n])
        assert modes.shape == (2, n)

        for i in range(5):
            start = n + i * step
            modes, cfreqs = proc.update(signal[start:start + step])
            assert modes.shape == (2, n)
            assert cfreqs.shape == (2,)

    def test_wrong_step_size_after_init(self):
        """Wrong number of samples after init with step_size>1 raises error."""
        n = 128
        step = 5
        proc = RSVMDProcessor(window_len=n, step_size=step, k=2)
        proc.update(np.zeros(n))
        with pytest.raises(ValueError, match=f"Expected {step} samples"):
            proc.update(np.zeros(1))


class TestWarmStartIterations:
    def test_warm_uses_fewer_iterations(self):
        """Warm frames should converge in fewer iterations than cold start."""
        n = 256
        signal = make_signal(n + 10, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        cold_iters = proc.last_iterations

        warm_iters = []
        for i in range(10):
            proc.update(signal[n + i : n + i + 1])
            warm_iters.append(proc.last_iterations)

        avg_warm = sum(warm_iters) / len(warm_iters)
        assert avg_warm <= cold_iters, (
            f"Avg warm iterations ({avg_warm:.1f}) should be <= cold start ({cold_iters})"
        )

    def test_last_converged_property(self):
        """last_converged is accessible after update."""
        n = 256
        signal = make_signal(n)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        proc.update(signal)
        # Should be a bool
        assert isinstance(proc.last_converged, bool)


class TestCenterFreqStability:
    def test_freqs_stay_near_true_values(self):
        """Center frequencies should stay near true values across streaming."""
        n = 256
        total = n + 20
        t = np.arange(total, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])

        expected = np.array([20.0 / n, 80.0 / n])
        tol = 15.0 / n

        for i in range(20):
            _, cfreqs = proc.update(signal[n + i : n + i + 1])
            sorted_freqs = np.sort(cfreqs)
            np.testing.assert_allclose(
                sorted_freqs, expected, atol=tol,
                err_msg=f"Frame {i}: frequencies drifted from true values",
            )


class TestLongStreaming:
    def test_200_frames_no_nan_or_inf(self):
        """200-frame streaming should not produce NaN or Inf."""
        n = 256
        total = n + 200
        signal = make_signal(total)

        proc = RSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(200):
            modes, cfreqs = proc.update(signal[n + i : n + i + 1])
            assert not np.any(np.isnan(modes)), f"NaN in modes at frame {i}"
            assert not np.any(np.isinf(modes)), f"Inf in modes at frame {i}"
            assert not np.any(np.isnan(cfreqs)), f"NaN in cfreqs at frame {i}"
            assert np.all(cfreqs >= 0), f"Negative center freq at frame {i}"

    def test_200_frames_center_freq_bounded(self):
        """Center frequencies stay bounded across 200 frames."""
        n = 256
        total = n + 200
        signal = make_signal(total, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(200):
            _, cfreqs = proc.update(signal[n + i : n + i + 1])
            # Frequencies should stay in [0, 0.5] (Nyquist)
            assert np.all(cfreqs >= 0), f"Negative freq at frame {i}"
            assert np.all(cfreqs <= 0.5), f"Freq > Nyquist at frame {i}"


class TestPaperClaims:
    def test_mode_spectral_separation(self):
        """Each mode should capture a distinct frequency band (VMD paper)."""
        n = 512
        t = np.arange(n, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 100 * t)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        modes, cfreqs = proc.update(signal)

        for ki in range(2):
            mode_fft = np.fft.fft(modes[ki])
            power = np.abs(mode_fft[:n // 2 + 1]) ** 2
            freqs = np.arange(n // 2 + 1, dtype=np.float64) / n

            omega_k = cfreqs[ki]
            bandwidth = 20.0 / n

            near_mask = np.abs(freqs - omega_k) < bandwidth
            near_power = power[near_mask].sum()
            total_power = power.sum()

            concentration = near_power / total_power if total_power > 1e-30 else 0
            assert concentration > 0.5, (
                f"Mode {ki} spectral concentration: {concentration:.4f} "
                f"(center={omega_k:.4f}), should be >0.5"
            )

    def test_mode_cross_correlation_low(self):
        """Modes from well-separated sinusoids should be nearly orthogonal."""
        n = 512
        t = np.arange(n, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 100 * t)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        modes, _ = proc.update(signal)

        dot = np.sum(modes[0] * modes[1])
        norm0 = np.linalg.norm(modes[0])
        norm1 = np.linalg.norm(modes[1])
        corr = abs(dot / (norm0 * norm1)) if norm0 > 1e-10 and norm1 > 1e-10 else 0

        assert corr < 0.3, f"Cross-correlation should be low: {corr:.4f}"

    def test_warm_converges_within_10(self):
        """Warm frames should converge quickly (paper claims 2-5 for tau=0)."""
        n = 512
        total = n + 20
        t = np.arange(total, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.0, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )
        proc.update(signal[:n])

        warm_iters = []
        for i in range(20):
            proc.update(signal[n + i : n + i + 1])
            warm_iters.append(proc.last_iterations)

        median = sorted(warm_iters)[len(warm_iters) // 2]
        assert median <= 10, (
            f"Median warm iterations should be ≤10, got {median} (all: {warm_iters})"
        )

    def test_rsvmd_matches_batch_vmd(self):
        """RSVMD streaming should match batch VMD on stationary signal."""
        n = 256
        total = n + 10
        t = np.arange(total, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)

        # RSVMD: cold start + 10 warm frames
        proc_stream = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )
        proc_stream.update(signal[:n])
        for i in range(10):
            proc_stream.update(signal[n + i : n + i + 1])
        rsvmd_freqs = np.sort(proc_stream.center_freqs())

        # Batch VMD on same final window
        proc_batch = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        proc_batch.update(signal[10:10 + n])
        batch_freqs = np.sort(proc_batch.center_freqs())

        tol = 10.0 / n
        np.testing.assert_allclose(
            rsvmd_freqs, batch_freqs, atol=tol,
            err_msg="RSVMD streaming should match batch VMD center frequencies",
        )


class TestFftResetInterval:
    def test_fft_reset_interval_produces_valid_output(self):
        """FFT reset at interval doesn't break streaming."""
        n = 128
        total = n + 20
        signal = make_signal(total)

        proc = RSVMDProcessor(
            alpha=2000.0, k=2, window_len=n, step_size=1,
            max_iter=500, fft_reset_interval=5,
        )

        proc.update(signal[:n])
        for i in range(20):
            modes, cfreqs = proc.update(signal[n + i:n + i + 1])
            assert modes.shape == (2, n)
            assert not np.any(np.isnan(modes))
