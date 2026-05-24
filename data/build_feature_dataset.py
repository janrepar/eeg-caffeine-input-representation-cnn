import numpy as np

from src.feature_extraction.extract_features import extract_features_2d_from_eeg
from src.utils.helpers import get_scalar, condition_to_label


def build_feature_dataset_from_alleeg(ALLEEG, fs: int = 600, include_subjects=None, feature_config: dict | None = None, exclude_subjects=None, include_conditions=None, include_groups=None):
    """
    Builds epoch-level feature dataset from ALLEEG.

    Parameters
    ----------
    ALLEEG: EEGLAB ALLEEG structure loaded from .mat file.
    fs: Sampling frequency.
    include_subjects: Optional list of subjects to include.
    exclude_subjects: Optional list of subjects to exclude.
    include_conditions: Optional list of conditions to include, e.g. ["Before", "After"].

    Returns
    -------
    X_2d: Shape: (n_total_epochs, n_channels, n_features_per_channel)
    y: Shape: (n_total_epochs) Before = 0, After = 1
    subjects: Shape: (n_total_epochs)
    conditions: Shape: (n_total_epochs)
    groups: (n_epochs)
    """

    if include_subjects is not None:
        include_subjects = set(map(str, include_subjects))

    if exclude_subjects is None:
        exclude_subjects = set()
    else:
        exclude_subjects = set(map(str, exclude_subjects))

    if include_conditions is not None:
        include_conditions = set(c.lower() for c in include_conditions)

    if include_groups is not None:
        include_groups = set(g.lower() for g in include_groups)

    X_list = []
    y_list = []
    subject_list = []
    condition_list = []
    group_list = []

    for i in range(ALLEEG.size):
        eeg = ALLEEG[0, i]

        subject = str(get_scalar(eeg["subject"]))
        condition = str(get_scalar(eeg["condition"]))
        group = str(get_scalar(eeg["group"]))

        if include_subjects is not None and subject not in include_subjects:
            continue

        if subject in exclude_subjects:
            continue

        if include_conditions is not None and condition.lower() not in include_conditions:
            continue

        if include_groups is not None and group.lower() not in include_groups:
            continue

        data = eeg["data"]

        if data is None or data.size == 0:
            print(f"Skipping empty data: subject={subject}, condition={condition}")
            continue

        label = condition_to_label(condition)

        X_features = extract_features_2d_from_eeg(
            data=data,
            fs=fs,
            feature_config=feature_config,
        )

        n_epochs = X_features.shape[0]

        X_list.append(X_features)
        y_list.extend([label] * n_epochs)
        subject_list.extend([subject] * n_epochs)
        condition_list.extend([condition] * n_epochs)
        group_list.extend([group] * n_epochs)

        print(f"{i + 1}. Subject={subject}, Group={group}, Condition={condition}, "f"Data={data.shape}, Features={X_features.shape}")

    if len(X_list) == 0:
        raise ValueError("No feature data found. Check filtering options.")

    X_2d = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list)
    conditions = np.array(condition_list)
    groups = np.array(group_list)

    return X_2d, y, subjects, conditions, groups