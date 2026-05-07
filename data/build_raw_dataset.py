import numpy as np

from utils.helpers import get_scalar, condition_to_label


def build_raw_dataset_from_alleeg(ALLEEG, exclude_subjects=None):
    """
    Builds epoch-level raw EEG dataset from ALLEEG.

    Returns:
        X_raw: (n_epochs, n_channels, n_timepoints)
        y: (n_epochs,)
        subjects: (n_epochs)
        conditions: (n_epochs)
    """

    if exclude_subjects is None:
        exclude_subjects = []

    X_list = []
    y_list = []
    subject_list = []
    condition_list = []

    for i in range(ALLEEG.size):
        eeg = ALLEEG[0, i]

        subject = str(get_scalar(eeg["subject"]))
        condition = str(get_scalar(eeg["condition"]))

        if subject in exclude_subjects:
            continue

        data = eeg["data"]  # (channels, timepoints, epochs)
        label = condition_to_label(condition)

        # move epochs to first dimension
        # from (channels, timepoints, epochs)
        # to (epochs, channels, timepoints)
        data_epochs = np.transpose(data, (2, 0, 1)).astype(np.float32)

        n_epochs = data_epochs.shape[0]

        X_list.append(data_epochs)
        y_list.extend([label] * n_epochs)
        subject_list.extend([subject] * n_epochs)
        condition_list.extend([condition] * n_epochs)

        print(
            f"{i+1}. Subject={subject}, Condition={condition}, "
            f"Input={data.shape}, Output={data_epochs.shape}"
        )

    X_raw = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list)
    conditions = np.array(condition_list)

    return X_raw, y, subjects, conditions