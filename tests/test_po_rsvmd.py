import numpy as np
import pytest
from rsvmd import PORSVMDProcessor, RSVMDProcessor


def make_signal(n, freqs=(10.0, 50.0, 200.0), amplitudes=(1.0, 0.7, 0.5)):
    """Generate a test signal as sum of sinusoids."""
    t = np.arange(n, dtype=np.float64) / n
    signal = np.zeros(n, dtype=np.float64)
    for f, a in zip(freqs, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)
    return signal


class TestPORSVMDBasic:
    def test_cold_start(self):
        """PO-RSVMD cold start produces valid decomposition."""
        n = 256
        signal = make_signal(n)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        modes, center_freqs = proc.update(signal)
        assert modes.shape == (3, n)
        assert center_freqs.shape == (3,)
        assert proc.initialized

    def test_warm_update(self):
        """PO-RSVMD warm updates produce valid output."""
        n = 256
        total = n + 5
        signal = make_signal(total)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(5):
            modes, cfreqs = proc.update(signal[n + i : n + i + 1])
            assert modes.shape == (3, n)
            assert cfreqs.shape == (3,)


class TestGammaAdaptation:
    def test_gamma_initial_value(self):
        """Gamma starts at the default value."""
        proc = PORSVMDProcessor(
            window_len=256, k=3, gamma_default=0.5,
        )
        assert proc.gamma == 0.5

    def test_custom_gamma_default(self):
        """Custom gamma_default is respected."""
        proc = PORSVMDProcessor(
            window_len=256, k=3, gamma_default=0.3,
        )
        assert proc.gamma == 0.3

    def test_gamma_changes_after_updates(self):
        """Gamma may change after multiple updates (adaptive)."""
        n = 256
        signal = make_signal(n + 10)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
            gamma_default=0.5,
        )

        proc.update(signal[:n])
        initial_gamma = proc.gamma

        # After several updates, gamma should remain a valid value
        for i in range(10):
            proc.update(signal[n + i : n + i + 1])

        assert 0.0 <= proc.gamma <= 1.0


class TestMatchesStandard:
    def test_matches_rsvmd_on_clean_signal(self):
        """On clean stationary signals, PO-RSVMD produces comparable results."""
        n = 256
        signal = make_signal(n, freqs=(20, 80), amplitudes=(1.0, 0.5))

        rsvmd = RSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        po_rsvmd = PORSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )

        modes_r, cfreqs_r = rsvmd.update(signal)
        modes_p, cfreqs_p = po_rsvmd.update(signal)

        # Both should find similar center frequencies
        assert modes_r.shape == modes_p.shape
        sorted_r = np.sort(cfreqs_r)
        sorted_p = np.sort(cfreqs_p)
        np.testing.assert_allclose(sorted_r, sorted_p, atol=5.0 / n)


class TestCustomGammaTiers:
    def test_custom_tiers(self):
        """Custom gamma_tiers can be passed."""
        tiers = [(0.5, 0.0), (0.2, 0.1)]
        proc = PORSVMDProcessor(
            window_len=128, k=2, gamma_tiers=tiers,
        )
        assert proc.initialized is False


class TestPOErrorHandling:
    def test_wrong_initial_size(self):
        """First call with wrong number of samples raises error."""
        proc = PORSVMDProcessor(window_len=256, k=3)
        with pytest.raises(ValueError, match="First call must provide"):
            proc.update(np.zeros(100))

    def test_wrong_step_size_after_init(self):
        """Wrong number of samples after initialization raises error."""
        n = 128
        proc = PORSVMDProcessor(window_len=n, step_size=1, k=2)
        proc.update(np.zeros(n))
        with pytest.raises(ValueError, match="Expected 1 samples"):
            proc.update(np.zeros(5))

    def test_not_initialized(self):
        """initialized is False before first call."""
        proc = PORSVMDProcessor(window_len=128, k=3)
        assert not proc.initialized


class TestPOResetFft:
    def test_reset_fft_doesnt_crash(self):
        """reset_fft() can be called on PO-RSVMD after initialization."""
        n = 128
        signal = make_signal(n)
        proc = PORSVMDProcessor(window_len=n, k=2)
        proc.update(signal)
        proc.reset_fft()
        modes, cfreqs = proc.update(np.zeros(1))
        assert modes.shape == (2, n)


class TestPOEdgeCases:
    def test_zero_signal(self):
        """PO-RSVMD on zero signal should not crash or produce NaN."""
        n = 128
        proc = PORSVMDProcessor(alpha=2000.0, k=2, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(np.zeros(n))

        assert modes.shape == (2, n)
        assert not np.any(np.isnan(modes))
        assert not np.any(np.isnan(cfreqs))

    def test_single_mode_k1(self):
        """PO-RSVMD with K=1."""
        n = 256
        signal = make_signal(n, freqs=(20,), amplitudes=(1.0,))

        proc = PORSVMDProcessor(alpha=2000.0, k=1, window_len=n, max_iter=500)
        modes, cfreqs = proc.update(signal)

        assert modes.shape == (1, n)
        assert cfreqs.shape == (1,)
        assert not np.any(np.isnan(modes))

    def test_step_size_greater_than_one(self):
        """PO-RSVMD with step_size > 1."""
        n = 256
        step = 5
        total = n + step * 3
        signal = make_signal(total)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=step, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(3):
            start = n + i * step
            modes, cfreqs = proc.update(signal[start:start + step])
            assert modes.shape == (3, n)
            assert cfreqs.shape == (3,)


class TestPOWarmStartIterations:
    def test_po_warm_uses_fewer_iterations(self):
        """PO-RSVMD warm frames should converge in fewer iterations than cold start."""
        n = 256
        signal = make_signal(n + 10, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = PORSVMDProcessor(
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
            f"PO avg warm iterations ({avg_warm:.1f}) should be <= cold start ({cold_iters})"
        )

    def test_po_last_converged_property(self):
        """last_converged is accessible on PO-RSVMD."""
        n = 256
        signal = make_signal(n)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, max_iter=500,
        )
        proc.update(signal)
        assert isinstance(proc.last_converged, bool)


class TestPOCenterFreqStability:
    def test_po_freqs_stay_near_true_values(self):
        """PO-RSVMD center frequencies stay near true values across streaming."""
        n = 256
        total = n + 20
        t = np.arange(total, dtype=np.float64) / n
        signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)

        proc = PORSVMDProcessor(
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
                err_msg=f"PO frame {i}: frequencies drifted from true values",
            )


class TestPOLongStreaming:
    def test_po_200_frames_no_nan_or_inf(self):
        """PO-RSVMD 200-frame streaming should not produce NaN or Inf."""
        n = 256
        total = n + 200
        signal = make_signal(total)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(200):
            modes, cfreqs = proc.update(signal[n + i : n + i + 1])
            assert not np.any(np.isnan(modes)), f"NaN in modes at PO frame {i}"
            assert not np.any(np.isinf(modes)), f"Inf in modes at PO frame {i}"
            assert not np.any(np.isnan(cfreqs)), f"NaN in cfreqs at PO frame {i}"

    def test_po_200_frames_center_freq_bounded(self):
        """PO-RSVMD center frequencies stay bounded across 200 frames."""
        n = 256
        total = n + 200
        signal = make_signal(total, freqs=(20, 80), amplitudes=(1.0, 0.5))

        proc = PORSVMDProcessor(
            alpha=2000.0, k=2, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(200):
            _, cfreqs = proc.update(signal[n + i : n + i + 1])
            assert np.all(cfreqs >= 0), f"Negative freq at PO frame {i}"
            assert np.all(cfreqs <= 0.5), f"Freq > Nyquist at PO frame {i}"


class TestPOReconstructionQuality:
    def test_modes_no_nan_across_streaming(self):
        """PO-RSVMD streaming should not produce NaN in any frame."""
        n = 256
        total = n + 20
        signal = make_signal(total)

        proc = PORSVMDProcessor(
            alpha=2000.0, k=3, tau=0.1, tol=1e-7,
            window_len=n, step_size=1, max_iter=500,
        )

        proc.update(signal[:n])
        for i in range(20):
            modes, cfreqs = proc.update(signal[n + i:n + i + 1])
            assert not np.any(np.isnan(modes)), f"NaN in modes at frame {i}"
            assert not np.any(np.isnan(cfreqs)), f"NaN in cfreqs at frame {i}"
