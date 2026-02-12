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
