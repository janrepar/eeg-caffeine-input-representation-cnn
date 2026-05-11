import numpy as np

from src.utils.helpers import get_scalar, condition_to_label


def build_raw_dataset_from_alleeg(
    ALLEEG,
    include_subjects=None,
    exclude_subjects=None,
    include_conditions=None,
):
    """
    Builds epoch-level raw EEG dataset from ALLEEG.

    Parameters
    ----------
    ALLEEG:
        EEGLAB ALLEEG structure loaded from .mat file.

    include_subjects:
        Optional list of subjects to include.

    exclude_subjects:
        Optional list of subjects to exclude.

    include_conditions:
        Optional list of conditions to include, e.g. ["Before", "After"].

    Returns
    -------
    X_raw:
        Shape: (n_total_epochs, n_channels, n_timepoints)

    y:
        Shape: (n_total_epochs,)
        Before = 0, After = 1

    subjects:
        Shape: (n_total_epochs,)

    conditions:
        Shape: (n_total_epochs,)
    """

    if include_subjects is not None:
        include_subjects = set(map(str, include_subjects))

    if exclude_subjects is None:
        exclude_subjects = set()
    else:
        exclude_subjects = set(map(str, exclude_subjects))

    if include_conditions is not None:
        include_conditions = set(c.lower() for c in include_conditions)

    X_list = []
    y_list = []
    subject_list = []
    condition_list = []

    for i in range(ALLEEG.size):
        eeg = ALLEEG[0, i]

        subject = str(get_scalar(eeg["subject"]))
        condition = str(get_scalar(eeg["condition"]))

        if include_subjects is not None and subject not in include_subjects:
            continue

        if subject in exclude_subjects:
            continue

        if include_conditions is not None and condition.lower() not in include_conditions:
            continue

        data = eeg["data"]

        if data is None or data.size == 0:
            print(f"Skipping empty data: subject={subject}, condition={condition}")
            continue

        label = condition_to_label(condition)

        # Original EEGLAB shape:
        # (channels, timepoints, epochs)
        #
        # PyTorch-friendly shape:
        # (epochs, channels, timepoints)
        data_epochs = np.transpose(data, (2, 0, 1)).astype(np.float32)

        n_epochs = data_epochs.shape[0]

        X_list.append(data_epochs)
        y_list.extend([label] * n_epochs)
        subject_list.extend([subject] * n_epochs)
        condition_list.extend([condition] * n_epochs)

        print(
            f"{i + 1}. Subject={subject}, Condition={condition}, "
            f"Input={data.shape}, Output={data_epochs.shape}"
        )

    if len(X_list) == 0:
        raise ValueError("No EEG data found. Check filtering options.")

    X_raw = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    subjects = np.array(subject_list)
    conditions = np.array(condition_list)

    return X_raw, y, subjects, conditions