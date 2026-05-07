import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}


def bandpower(signal: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)

    if not np.any(idx):
        return 0.0

    return float(trapezoid(psd[idx], freqs[idx]))


def extract_frequency_features(signal: np.ndarray, fs: int) -> list[float]:
    return [
        bandpower(signal, fs, band_range)
        for band_range in EEG_BANDS.values()
    ]


FREQUENCY_FEATURE_NAMES = [
    f"{band_name}_power"
    for band_name in EEG_BANDS.keys()
]