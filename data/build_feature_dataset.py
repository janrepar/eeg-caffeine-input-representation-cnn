import numpy as np

from src.feature_extraction.extract_features import extract_features_2d_from_eeg
from utils.helpers import get_scalar, condition_to_label


def build_feature_dataset_from_alleeg(ALLEEG, fs: int = 600):
    """
    ALLEEG vsebuje več zapisov oblike:
    subject + condition + data

    data shape: (n_channels, n_timepoints, n_epochs)

    Returns:
        X_2d:       (n_total_epochs, n_channels, n_features_per_channel)
        y:          (n_total_epochs,)
        subjects:   (n_total_epochs,)
        conditions: (n_total_epochs,)
    """

    X_list = []
    y_list = []
    subject_list = []
    condition_list = []

    for i in range(ALLEEG.size):
        eeg = ALLEEG[0, i]

        subject = str(get_scalar(eeg["subject"]))
        condition = str(get_scalar(eeg["condition"]))
        data = eeg["data"]

        label = condition_to_label(condition)

        # data: (channels, timepoints, epochs)
        X_features = extract_features_2d_from_eeg(data, fs)

        n_epochs = X_features.shape[0]

        X_list.append(X_features)
        y_list.extend([label] * n_epochs)
        subject_list.extend([subject] * n_epochs)
        condition_list.extend([condition] * n_epochs)

        print(
            f"{i+1}. Subject={subject}, Condition={condition}, "
            f"Data={data.shape}, Features={X_features.shape}"
        )

    X_2d = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list)
    conditions = np.array(condition_list)

    return X_2d, y, subjects, conditions