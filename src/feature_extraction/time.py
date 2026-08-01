import numpy as np
from scipy.stats import skew, kurtosis


def extract_time_features(signal: np.ndarray) -> list[float]:
    _, hjorth_mobility, hjorth_complexity = hjorth_parameters(signal)

    return [
        float(np.mean(signal)),
        float(np.std(signal)),
        float(skew(signal)),
        float(kurtosis(signal, fisher=False)),
        float(np.min(signal)),
        float(np.max(signal)),
        float(hjorth_mobility),
        float(hjorth_complexity),
    ]


def hjorth_parameters(signal: np.ndarray):
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero

    mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0

    mobility_d1 = np.sqrt(var_d2 / var_d1) if var_d1 != 0 else 0

    complexity = mobility_d1 / mobility if mobility != 0 else 0

    return activity, mobility, complexity


STATISTICAL_FEATURE_NAMES = [
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "min",
    "max",
    "hjorth_mobility",
    "hjorth_complexity"
]