import numpy as np
from sklearn.model_selection import GroupKFold


def create_loso_splits(subjects: np.ndarray):
    """
    Creates Leave-One-Subject-Out splits.

    For each fold:
        test = one subject
        train = all remaining subjects

    Parameters
    ----------
    subjects: Array with subject ID for each epoch. Shape: (n_epochs)

    Returns
    -------
    splits: List of dictionaries with: test_subject, train_mask, test_mask
    """

    unique_subjects = np.unique(subjects)
    splits = []

    for test_subject in unique_subjects:
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        splits.append({
            "test_subject": test_subject,
            "train_mask": train_mask,
            "test_mask": test_mask
        })

    return splits


def create_train_val_split_by_subject(
    subjects: np.ndarray,
    train_mask: np.ndarray,
    random_seed: int = 42,
    n_validation_subjects=1
):
    """
    Creates validation split from training subjects only.

    N subjects from the training set is selected as a validation subject.
    This avoids using the LOSO test subject for early stopping.

    Parameters
    ----------
    subjects: Subject ID for each epoch.
    train_mask: Boolean mask indicating which epochs are available for training.
    random_seed: Seed for reproducible validation subject selection.

    Returns
    -------
    final_train_mask: Boolean mask for final training epochs.
    val_mask: Boolean mask for validation epochs.
    val_subject: Subject selected for validation.
    """

    rng = np.random.default_rng(random_seed)

    available_train_subjects = np.unique(subjects[train_mask])

    if len(available_train_subjects) <= n_validation_subjects:
        raise ValueError(
            f"Not enough training subjects for validation split. "
            f"Available train subjects: {len(available_train_subjects)}, "
            f"requested validation subjects: {n_validation_subjects}"
        )

    validation_subjects = rng.choice(
        available_train_subjects,
        size=n_validation_subjects,
        replace=False,
    )

    val_mask = train_mask & np.isin(subjects, validation_subjects)
    new_train_mask = train_mask & ~np.isin(subjects, validation_subjects)

    return new_train_mask, val_mask, validation_subjects


def create_groupkfold_splits(subjects, n_splits=5):
    """
    Creates subject-independent GroupKFold splits.

    All epochs from the same subject are kept together.
    Each subject appears in the test set exactly once across folds.
    """

    subjects = np.asarray(subjects)

    dummy_X = np.zeros(len(subjects))
    dummy_y = np.zeros(len(subjects))

    group_kfold = GroupKFold(n_splits=n_splits)

    splits = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        group_kfold.split(dummy_X, dummy_y, groups=subjects),
        start=1,
    ):
        train_mask = np.zeros(len(subjects), dtype=bool)
        test_mask = np.zeros(len(subjects), dtype=bool)

        train_mask[train_idx] = True
        test_mask[test_idx] = True

        test_subjects = np.unique(subjects[test_mask])

        splits.append(
            {
                "fold": fold_idx,
                "train_mask": train_mask,
                "test_mask": test_mask,
                "test_subject": ",".join(map(str, test_subjects)),
                "test_subjects": test_subjects,
            }
        )

    return splits


def print_split_summary(subjects, y, train_mask, val_mask, test_mask):
    """
    Prints basic split information for debugging.
    """

    print("Split summary:")
    print(f"Train epochs: {train_mask.sum()}")
    print(f"Val epochs:   {val_mask.sum()}")
    print(f"Test epochs:  {test_mask.sum()}")

    print("\nTrain subjects:", np.unique(subjects[train_mask]))
    print("Val subjects:", np.unique(subjects[val_mask]))
    print("Test subjects:", np.unique(subjects[test_mask]))

    print("\nClass distribution:")
    print(f"Train Before: {(y[train_mask] == 0).sum()}, After: {(y[train_mask] == 1).sum()}")
    print(f"Val Before:   {(y[val_mask] == 0).sum()}, After: {(y[val_mask] == 1).sum()}")
    print(f"Test Before:  {(y[test_mask] == 0).sum()}, After: {(y[test_mask] == 1).sum()}")