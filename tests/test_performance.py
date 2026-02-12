"""Performance test for RSVMD release build.

Run manually: python tests/test_performance.py
Requires: maturin develop --release
"""

import time

import numpy as np

from rsvmd import RSVMDProcessor

N = 7200
K = 3
N_WARM_FRAMES = 100


def make_signal(n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / n
    return (
        np.sin(2 * np.pi * 10.0 * t)
        + 0.7 * np.sin(2 * np.pi * 50.0 * t)
        + 0.5 * np.sin(2 * np.pi * 200.0 * t)
    )


def main() -> None:
    signal = make_signal(N + N_WARM_FRAMES)

    proc = RSVMDProcessor(
        alpha=2000.0,
        k=K,
        tau=0.0,
        tol=1e-7,
        window_len=N,
        step_size=1,
        max_iter=500,
    )

    # Cold start timing
    t0 = time.perf_counter()
    proc.update(signal[:N])
    cold_ms = (time.perf_counter() - t0) * 1000

    # Warm frame timings
    warm_times = []
    for i in range(N_WARM_FRAMES):
        t0 = time.perf_counter()
        proc.update(signal[N + i : N + i + 1])
        warm_times.append((time.perf_counter() - t0) * 1000)

    warm_arr = np.array(warm_times)
    median_warm = float(np.median(warm_arr))
    p95_warm = float(np.percentile(warm_arr, 95))

    print(f"Cold start:         {cold_ms:.2f} ms")
    print(f"Warm frame median:  {median_warm:.2f} ms")
    print(f"Warm frame p95:     {p95_warm:.2f} ms")
    print(f"Warm frame min:     {float(warm_arr.min()):.2f} ms")
    print(f"Warm frame max:     {float(warm_arr.max()):.2f} ms")

    assert cold_ms < 500, f"Cold start too slow: {cold_ms:.1f} ms (limit 500 ms)"
    assert median_warm < 50, f"Median warm frame too slow: {median_warm:.1f} ms (limit 50 ms)"

    print("\nPASS")


if __name__ == "__main__":
    main()
