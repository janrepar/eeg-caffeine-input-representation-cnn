import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Make project root available for imports
sys.path.append(os.path.abspath(""))

from src.utils.helpers import load_config
from data.build_raw_dataset import build_raw_dataset_from_alleeg
from data.build_feature_dataset import build_feature_dataset_from_alleeg


def main():
    config = load_config("config.yaml")

    data_path = config["data"]["data_path"]
    fs = config["data"]["sampling_rate"]
    include_conditions = config["data"].get("include_conditions", None)
    exclude_subjects = config["data"].get("exclude_subjects", [])
    feature_config = config.get("features", None)
    raw_dataset_output = config["data"]["raw_dataset_output"]
    feature_dataset_output = config["data"]["feature_dataset_output"]

    mat = loadmat(data_path, simplify_cells=False)
    ALLEEG = mat["ALLEEG"]

    X_raw, y_raw, subjects_raw, conditions_raw, groups_raw = build_raw_dataset_from_alleeg(
        ALLEEG,
        include_conditions=include_conditions,
        exclude_subjects=exclude_subjects,
    )

    X_feat, y_feat, subjects_feat, conditions_feat, groups_feat = build_feature_dataset_from_alleeg(
        ALLEEG,
        fs=fs,
        feature_config=feature_config,
        include_conditions=include_conditions,
        exclude_subjects=exclude_subjects,
    )

    print("\nRAW DATASET")
    print("X_raw:", X_raw.shape)
    print("y_raw:", y_raw.shape)
    print("subjects_raw:", subjects_raw.shape)
    print("conditions_raw:", conditions_raw.shape)
    print("groups_raw:", groups_raw.shape)

    print("\nFEATURE DATASET")
    print("X_feat:", X_feat.shape)
    print("y_feat:", y_feat.shape)
    print("subjects_feat:", subjects_feat.shape)
    print("conditions_feat:", conditions_feat.shape)
    print("groups_feat:", groups_feat.shape)

    print("\nCHECKING ALIGNMENT")
    print("Labels equal:", np.array_equal(y_raw, y_feat))
    print("Subjects equal:", np.array_equal(subjects_raw, subjects_feat))
    print("Conditions equal:", np.array_equal(conditions_raw, conditions_feat))
    print("Groups equal:", np.array_equal(groups_raw, groups_feat))

    assert np.array_equal(y_raw, y_feat), "Labels do not match!"
    assert np.array_equal(subjects_raw, subjects_feat), "Subjects do not match!"
    assert np.array_equal(conditions_raw, conditions_feat), "Conditions do not match!"
    assert np.array_equal(groups_raw, groups_feat), "Groups do not match!"

    print("\nDataset builders are aligned.")

    unique_subjects = np.unique(subjects_raw)
    print("\nNumber of subjects:", len(unique_subjects))
    print("Subjects:", unique_subjects)

    print("\nLabel distribution:")
    print("Before:", np.sum(y_raw == 0))
    print("After:", np.sum(y_raw == 1))

    print("\nGroup distribution:")
    print("Caffeine:", np.sum(groups_raw == "Caffeine"))
    print("Placebo:", np.sum(groups_raw == "Placebo"))

    print("\nSubjects by group:")
    for group in np.unique(groups_raw):
        group_subjects = np.unique(subjects_raw[groups_raw == group])
        print(f"{group}: {len(group_subjects)} subjects -> {group_subjects}")

    save_dataset_npz(
        output_path=raw_dataset_output,
        X=X_raw,
        y=y_raw,
        subjects=subjects_raw,
        conditions=conditions_raw,
        groups = groups_raw
    )

    save_dataset_npz(
        output_path=feature_dataset_output,
        X=X_feat,
        y=y_feat,
        subjects=subjects_feat,
        conditions=conditions_feat,
        groups=groups_feat
    )


def save_dataset_npz(output_path, X, y, subjects, conditions, groups):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        subjects=subjects,
        conditions=conditions,
        groups=groups
    )

    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    main()