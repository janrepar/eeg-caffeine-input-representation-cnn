import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def spectral_entropy(signal: np.ndarray, fs: int) -> float:
    _, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    psd = psd + 1e-12
    psd_norm = psd / np.sum(psd)

    return entropy(psd_norm)


def extract_entropy_features(signal: np.ndarray, fs: int) -> list[float]:
    return [
        spectral_entropy(signal, fs)
    ]

ENTROPY_FEATURE_NAMES = [
    "spectral_entropy",
]