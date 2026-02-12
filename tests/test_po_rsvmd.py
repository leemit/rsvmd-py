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

    def test_not_initialized(self):
        """initialized is False before first call."""
        proc = PORSVMDProcessor(window_len=128, k=3)
        assert not proc.initialized
