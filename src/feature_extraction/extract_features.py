import numpy as np

from src.feature_extraction.time import (
    extract_time_features,
    STATISTICAL_FEATURE_NAMES,
)
from src.feature_extraction.frequency import (
    extract_frequency_features,
    get_frequency_feature_names,
)
from src.feature_extraction.entropy import (
    extract_entropy_features,
    ENTROPY_FEATURE_NAMES,
)


def get_feature_names_per_channel(feature_config: dict | None = None) -> list[str]:
    """
    Returns feature names based on configuration.
    """

    if feature_config is None:
        feature_config = {
            "use_time": True,
            "use_frequency": True,
            "use_entropy": True,
            "frequency_bands": {
                "theta": [4, 8],
                "alpha": [8, 13],
                "beta": [13, 30],
                "gamma": [30, 45],
            },
        }

    feature_names = []

    if feature_config.get("use_time", True):
        feature_names.extend(STATISTICAL_FEATURE_NAMES)

    if feature_config.get("use_frequency", True):
        frequency_bands = feature_config.get("frequency_bands", None)
        feature_names.extend(get_frequency_feature_names(frequency_bands))

    if feature_config.get("use_entropy", True):
        feature_names.extend(ENTROPY_FEATURE_NAMES)

    return feature_names


def extract_channel_features(signal: np.ndarray, fs: int, feature_config: dict | None = None) -> list[float]:
    """
    Extracts selected features from one EEG channel signal.

    signal shape: (n_timepoints)
    """

    if feature_config is None:
        feature_config = {
            "use_time": True,
            "use_frequency": True,
            "use_entropy": True,
            "frequency_bands": {
                "theta": [4, 8],
                "alpha": [8, 13],
                "beta": [13, 30],
                "gamma": [30, 45],
            },
        }

    features = []

    if feature_config.get("use_time", True):
        features.extend(extract_time_features(signal))

    if feature_config.get("use_frequency", True):
        frequency_bands = feature_config.get("frequency_bands", None)
        features.extend(extract_frequency_features(signal, fs, frequency_bands, nperseg=feature_config.get("welch_nperseg")))

    if feature_config.get("use_entropy", True):
        features.extend(extract_entropy_features(signal, fs, nperseg=feature_config.get("welch_nperseg")))

    return features


def extract_epoch_features(epoch: np.ndarray, fs: int, feature_config: dict | None = None) -> np.ndarray:
    """
    Extracts features from one EEG epoch.

    epoch shape: (n_channels, n_timepoints)

    returns: (n_channels * n_features_per_channel)
    """

    epoch_features = []

    for channel_idx in range(epoch.shape[0]):
        channel_signal = epoch[channel_idx, :]
        channel_features = extract_channel_features(
            signal=channel_signal,
            fs=fs,
            feature_config=feature_config
        )
        epoch_features.extend(channel_features)

    return np.array(epoch_features, dtype=np.float32)


def extract_features_from_eeg(data: np.ndarray, fs: int, feature_config: dict | None = None) -> np.ndarray:
    """
    Extracts flattened feature vectors from the full EEG dataset.

    data shape: (n_channels, n_timepoints, n_epochs)

    returns: X_features shape: (n_epochs, n_channels * n_features_per_channel)
    """

    _, _, n_epochs = data.shape

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch = data[:, :, epoch_idx]
        features = extract_epoch_features(
            epoch=epoch,
            fs=fs,
            feature_config=feature_config
        )
        all_features.append(features)

    return np.array(all_features, dtype=np.float32)


def extract_features_2d_from_eeg(data: np.ndarray, fs: int, feature_config: dict | None = None) -> np.ndarray:
    """
    Extracts 2D feature representation from the full EEG dataset.

    data shape: (n_channels, n_timepoints, n_epochs)

    returns: X_features_2d shape: (n_epochs, n_channels, n_features_per_channel)
    """

    n_channels, _, n_epochs = data.shape
    feature_names = get_feature_names_per_channel(feature_config)
    n_features = len(feature_names)

    all_features = np.zeros(
        (n_epochs, n_channels, n_features),
        dtype=np.float32,
    )

    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            signal = data[channel_idx, :, epoch_idx]
            all_features[epoch_idx, channel_idx, :] = extract_channel_features(
                signal=signal,
                fs=fs,
                feature_config=feature_config
            )

    return all_features
