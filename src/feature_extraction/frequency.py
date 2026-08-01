import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

DEFAULT_EEG_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

def normalize_frequency_bands(frequency_bands=None):
    if frequency_bands is None:
        return DEFAULT_EEG_BANDS
    return {name: tuple(value) for name, value in frequency_bands.items()}

def get_frequency_feature_names(frequency_bands=None) -> list[str]:
    return [f"{name}_log_power" for name in normalize_frequency_bands(frequency_bands)]

def bandpower(signal: np.ndarray, fs: int, band: tuple[float, float], nperseg: int | None = None) -> float:
    """Returns log10 absolute band power estimated with Welch's method."""
    if nperseg is None:
        nperseg = len(signal)
    freqs, psd = welch(signal, fs=fs, nperseg=min(int(nperseg), len(signal)))
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    if np.count_nonzero(idx) < 2:
        raise ValueError(f"Band {band} is not resolvable with nperseg={nperseg} at fs={fs}.")
    return float(np.log10(trapezoid(psd[idx], freqs[idx]) + 1e-12))

def extract_frequency_features(signal: np.ndarray, fs: int, frequency_bands=None, nperseg: int | None = None) -> list[float]:
    return [bandpower(signal, fs, band, nperseg=nperseg) for band in normalize_frequency_bands(frequency_bands).values()]