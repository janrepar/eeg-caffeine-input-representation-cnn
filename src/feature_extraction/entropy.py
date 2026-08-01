import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

def spectral_entropy(signal: np.ndarray, fs: int, nperseg: int | None = None) -> float:
    if nperseg is None:
        nperseg = len(signal)
    _, psd = welch(signal, fs=fs, nperseg=min(int(nperseg), len(signal)))
    psd = psd + 1e-12
    return entropy(psd / np.sum(psd))

def extract_entropy_features(signal: np.ndarray, fs: int, nperseg: int | None = None) -> list[float]:
    return [spectral_entropy(signal, fs, nperseg=nperseg)]

ENTROPY_FEATURE_NAMES = ["spectral_entropy"]