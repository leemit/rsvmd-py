import numpy as np
import numpy.typing as npt

class RSVMDProcessor:
    @property
    def initialized(self) -> bool: ...
    @property
    def last_iterations(self) -> int: ...
    @property
    def last_converged(self) -> bool: ...
    def __init__(
        self,
        alpha: float = 2000.0,
        k: int = 3,
        tau: float = 0.0,
        tol: float = 1e-7,
        window_len: int = 7200,
        step_size: int = 1,
        max_iter: int = 500,
        damping: float = 0.99999,
        fft_reset_interval: int = 0,
    ) -> None: ...
    def update(
        self,
        samples: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def center_freqs(self) -> npt.NDArray[np.float64]: ...
    def reset_fft(self) -> None: ...

class PORSVMDProcessor:
    @property
    def initialized(self) -> bool: ...
    @property
    def gamma(self) -> float: ...
    @property
    def last_iterations(self) -> int: ...
    @property
    def last_converged(self) -> bool: ...
    def __init__(
        self,
        alpha: float = 2000.0,
        k: int = 3,
        tau: float = 0.0,
        tol: float = 1e-7,
        window_len: int = 7200,
        step_size: int = 1,
        max_iter: int = 500,
        damping: float = 0.99999,
        fft_reset_interval: int = 0,
        gamma_default: float = 0.5,
        gamma_tiers: list[tuple[float, float]] | None = None,
    ) -> None: ...
    def update(
        self,
        samples: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def center_freqs(self) -> npt.NDArray[np.float64]: ...
    def reset_fft(self) -> None: ...
