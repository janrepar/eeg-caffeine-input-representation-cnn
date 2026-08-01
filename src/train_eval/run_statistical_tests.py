"""Reference baselines and paired exact permutation tests for LOSO and GroupKFold."""
import os
import sys
from itertools import combinations, product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.train_eval.common import load_dataset_npz
from src.utils.helpers import load_config

MODELS = {
    "Raw EEGNet-like CNN": "raw_eegnetlike",
    "Features1D CNN": "features1d_cnn",
    "Features2D CNN": "features2d_cnn",
    "Hybrid Raw + Features CNN": "hybrid_cnn",
}


def latest_result_csv(results_root: Path, prefix: str, validation_method: str) -> Path:
    filename = f"{prefix}_{validation_method}_metrics.csv"
    paths = sorted(
        (
            path / filename
            for path in results_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix) and (path / filename).is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(
            f"No {validation_method.upper()} metrics found for {prefix} in {results_root}."
        )
    return paths[0]


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(differences.mean())
    null = np.fromiter(
        (
            abs(np.mean(differences * np.asarray(signs)))
            for signs in product((-1, 1), repeat=len(differences))
        ),
        dtype=float,
    )
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (len(null) + 1))


def baseline_metrics(true_labels: np.ndarray, fold: int, test_subjects: str) -> list[dict]:
    rows = []
    for name, value in (("Always Before", 0), ("Always After", 1)):
        predicted = np.full_like(true_labels, value)
        rows.append(
            {
                "model_name": name,
                "fold": fold,
                "test_subjects": test_subjects,
                "epoch_accuracy": accuracy_score(true_labels, predicted),
                "balanced_accuracy": balanced_accuracy_score(true_labels, predicted),
                "epoch_f1": f1_score(true_labels, predicted, zero_division=0),
            }
        )
    return rows


def make_baselines(y: np.ndarray, subjects: np.ndarray, reference: pd.DataFrame, validation_method: str) -> pd.DataFrame:
    """Build reference baseline scores on the exact test set of every evaluation unit."""
    subject_values = subjects.astype(str)
    rows = []
    for _, fold_row in reference.iterrows():
        if validation_method == "loso":
            subject_list = [str(fold_row["test_subject"])]
            test_label = subject_list[0]
        else:
            subject_list = [value.strip() for value in str(fold_row["test_subjects"]).split(",")]
            test_label = ",".join(subject_list)
        true_labels = y[np.isin(subject_values, subject_list)]
        if len(true_labels) == 0:
            raise ValueError(f"No epochs matched GroupKFold/LOSO test set: {test_label}")
        rows.extend(baseline_metrics(true_labels, int(fold_row["fold"]), test_label))
    return pd.DataFrame(rows)


def main():
    config = load_config("config.yaml")
    validation_method = str(config["validation"].get("method", "LOSO")).lower()
    if validation_method not in {"loso", "groupkfold"}:
        raise ValueError(f"Unsupported validation method for statistical tests: {validation_method}")

    experiment_dir_value = os.environ.get("EEG_EXPERIMENT_DIR")
    experiment_dir = Path(experiment_dir_value).resolve() if experiment_dir_value else None
    results_root = experiment_dir / "results" if experiment_dir is not None else Path(config["outputs"]["results_dir"])
    statistics_dir = experiment_dir / "statistical_tests" if experiment_dir is not None else results_root
    statistics_dir.mkdir(parents=True, exist_ok=True)

    _, y, subjects, _, groups = load_dataset_npz(config["data"]["raw_dataset_output"])
    analysis_type = config.get("experiment", {}).get("analysis_type", "before_vs_after")
    if analysis_type == "caffeine_before_vs_after":
        mask = np.char.lower(groups.astype(str)) == "caffeine"
    elif analysis_type == "placebo_before_vs_after":
        mask = np.char.lower(groups.astype(str)) == "placebo"
    elif analysis_type == "before_vs_after":
        mask = np.ones(len(y), dtype=bool)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")
    y, subjects = y[mask], subjects[mask]

    frames = []
    for model_name, prefix in MODELS.items():
        frame = pd.read_csv(latest_result_csv(results_root, prefix, validation_method))
        if "fold" not in frame.columns:
            raise ValueError(f"Missing fold column for {model_name}.")
        required_metrics = {"epoch_accuracy", "epoch_f1"}
        missing_metrics = required_metrics.difference(frame.columns)
        if missing_metrics:
            raise ValueError(f"Missing metrics for {model_name}: {sorted(missing_metrics)}")
        if validation_method == "loso" and "test_subject" not in frame.columns:
            raise ValueError(f"LOSO results for {model_name} need a test_subject column.")
        if validation_method == "groupkfold" and "test_subjects" not in frame.columns:
            raise ValueError(f"GroupKFold results for {model_name} need a test_subjects column.")
        frame["model_name"] = model_name
        frames.append(frame)

    reference = frames[0].sort_values("fold").reset_index(drop=True)
    reference_folds = reference["fold"].tolist()
    for frame in frames[1:]:
        if frame.sort_values("fold")["fold"].tolist() != reference_folds:
            raise ValueError("Models do not contain the same evaluation folds.")

    baselines = make_baselines(y, subjects, reference, validation_method)
    baselines.to_csv(statistics_dir / f"{validation_method}_reference_baselines.csv", index=False)

    scores = pd.concat(frames, ignore_index=True)
    unit_label = "participants" if validation_method == "loso" else "GroupKFold folds"
    test_description = (
        "exact sign-flip permutation across participants"
        if validation_method == "loso"
        else "exact sign-flip permutation across GroupKFold folds (exploratory; training sets overlap)"
    )
    rows = []

    for model_name, frame in scores.groupby("model_name", sort=False):
        values = frame.sort_values("fold")["epoch_accuracy"].to_numpy()
        rows.append(
            {
                "metric": "epoch_accuracy",
                "comparison": f"{model_name} vs chance (0.5)",
                "test": test_description,
                "mean_difference": float(np.mean(values - 0.5)),
                "p_value_two_sided": exact_sign_flip_pvalue(values - 0.5),
                "n_units": len(values),
                "unit": unit_label,
            }
        )

    for metric in ("epoch_accuracy", "epoch_f1"):
        pivot = scores.pivot(index="fold", columns="model_name", values=metric)
        for first, second in combinations(MODELS, 2):
            difference = pivot[first].to_numpy() - pivot[second].to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{first} vs {second}",
                    "test": test_description,
                    "mean_difference": float(difference.mean()),
                    "p_value_two_sided": exact_sign_flip_pvalue(difference),
                    "n_units": len(difference),
                    "unit": unit_label,
                }
            )

    pd.DataFrame(rows).to_csv(statistics_dir / f"{validation_method}_statistical_tests.csv", index=False)
    with open(statistics_dir / "README.txt", "w", encoding="utf-8") as output:
        output.write(f"Validation method: {validation_method.upper()}\n")
        output.write(f"Evaluation unit: {unit_label}\n")
        output.write(f"Test: {test_description}\n")
        if validation_method == "groupkfold":
            output.write("Interpret GroupKFold fold-level p-values as exploratory because training sets overlap.\n")

    print(f"Saved baselines: {statistics_dir / f'{validation_method}_reference_baselines.csv'}")
    print(f"Saved statistical tests: {statistics_dir / f'{validation_method}_statistical_tests.csv'}")


if __name__ == "__main__":
    main()