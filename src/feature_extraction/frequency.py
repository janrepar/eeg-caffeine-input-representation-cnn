import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


DEFAULT_EEG_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def normalize_frequency_bands(frequency_bands=None):
    if frequency_bands is None:
        return DEFAULT_EEG_BANDS

    return {
        band_name: tuple(band_range)
        for band_name, band_range in frequency_bands.items()
    }


def get_frequency_feature_names(frequency_bands=None) -> list[str]:
    bands = normalize_frequency_bands(frequency_bands)

    return [
        f"{band_name}_power"
        for band_name in bands.keys()
    ]


def bandpower(signal: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)

    if not np.any(idx):
        return 0.0

    return trapezoid(psd[idx], freqs[idx])


def extract_frequency_features(
    signal: np.ndarray,
    fs: int,
    frequency_bands=None,
) -> list[float]:
    bands = normalize_frequency_bands(frequency_bands)

    return [
        bandpower(signal, fs, band_range)
        for band_range in bands.values()
    ]