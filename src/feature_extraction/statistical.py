import numpy as np
from scipy.stats import skew, kurtosis


def extract_statistical_features(signal: np.ndarray) -> list[float]:
    return [
        float(np.mean(signal)),
        float(np.std(signal)),
        float(np.var(signal)),
        float(skew(signal)),
        float(kurtosis(signal, fisher=False)),
        float(np.min(signal)),
        float(np.max(signal)),
    ]


STATISTICAL_FEATURE_NAMES = [
    "mean",
    "std",
    "var",
    "skewness",
    "kurtosis",
    "min",
    "max",
]