import numpy as np

from src.feature_extraction.statistical import (
    extract_statistical_features,
    STATISTICAL_FEATURE_NAMES,
)
from src.feature_extraction.frequency import (
    extract_frequency_features,
    FREQUENCY_FEATURE_NAMES,
)
from src.feature_extraction.entropy import (
    extract_entropy_features,
    ENTROPY_FEATURE_NAMES,
)


FEATURE_NAMES_PER_CHANNEL = (
    STATISTICAL_FEATURE_NAMES
    + FREQUENCY_FEATURE_NAMES
    + ENTROPY_FEATURE_NAMES
)


def extract_channel_features(signal: np.ndarray, fs: int) -> list[float]:
    features = []

    features.extend(extract_statistical_features(signal))
    features.extend(extract_frequency_features(signal, fs))
    features.extend(extract_entropy_features(signal, fs))

    return features


def extract_epoch_features(epoch: np.ndarray, fs: int) -> np.ndarray:
    """
    epoch shape: (n_channels, n_timepoints)
    """
    epoch_features = []

    for channel_idx in range(epoch.shape[0]):
        channel_signal = epoch[channel_idx, :]
        channel_features = extract_channel_features(channel_signal, fs)
        epoch_features.extend(channel_features)

    return np.array(epoch_features, dtype=np.float32)


def extract_features_from_eeg(data: np.ndarray, fs: int) -> np.ndarray:
    """
    data shape: (n_channels, n_timepoints, n_epochs)

    returns:
        X_features shape: (n_epochs, n_channels * n_features_per_channel)
    """
    n_channels, _, n_epochs = data.shape

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch = data[:, :, epoch_idx]
        features = extract_epoch_features(epoch, fs)
        all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def extract_features_2d_from_eeg(data: np.ndarray, fs: int) -> np.ndarray:
    """
    data shape: (n_channels, n_timepoints, n_epochs)

    returns:
        X_features_2d shape: (n_epochs, n_channels, n_features_per_channel)
    """
    n_channels, _, n_epochs = data.shape
    n_features = len(FEATURE_NAMES_PER_CHANNEL)

    all_features = np.zeros((n_epochs, n_channels, n_features), dtype=np.float32)

    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            signal = data[channel_idx, :, epoch_idx]
            all_features[epoch_idx, channel_idx, :] = extract_channel_features(signal, fs)

    return all_features