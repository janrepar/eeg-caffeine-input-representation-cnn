import csv
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import shutil
import random
import torch


def set_random_seed(seed: int) -> None:
    """Sets random seeds for reproducible model training and data loading."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_dataset_npz(path):
    """
    Loads dataset saved as .npz.
    Expected keys: X, y, subjects, conditions
    """

    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    conditions = data["conditions"]
    groups = data["groups"] if "groups" in data.files else None

    return X, y, subjects, conditions, groups


def get_device(config):
    """
    Returns torch device based on config.
    """

    device_config = config["training"].get("device", "auto")

    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_config)



def print_model_parameter_count(model, model_name: str) -> None:
    """Print total and trainable parameter counts for a PyTorch model."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"{model_name} parameters: {total_parameters:,} total; "
        f"{trainable_parameters:,} trainable"
    )


def ensure_output_dirs(config):
    """
    Creates output directories and returns paths.
    """

    outputs_config = config["outputs"]

    models_dir = Path(outputs_config["models_dir"])
    results_dir = Path(outputs_config["results_dir"])
    plots_dir = Path(outputs_config["plots_dir"])

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, results_dir, plots_dir


def save_config_copy(config_path, output_dir):
    """
    Saves a copy of config.yaml into the experiment output folder.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, output_dir / "config_used.yaml")


EXPERIMENT_DIR_ENV = "EEG_EXPERIMENT_DIR"


def create_experiment_root(config, timestamp: str | None = None) -> Path:
    """Create the common output root for a complete experiment."""
    outputs_config = config["outputs"]
    validation_method = str(config["validation"].get("method", "LOSO")).upper()
    n_validation_subjects = config["validation"].get("n_validation_subjects", 1)
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiments_dir = Path(
        outputs_config.get(
            "experiments_dir",
            Path(outputs_config.get("output_dir", "outputs")) / "experiments",
        )
    )
    experiment_dir = experiments_dir / f"{timestamp}_{validation_method}_nvalsubj{n_validation_subjects}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    for directory_name in ("models", "results", "plots", "analysis", "experiment_timings", "statistical_tests"):
        (experiment_dir / directory_name).mkdir()

    config_path = Path("config.yaml")
    if config_path.is_file():
        save_config_copy(config_path, experiment_dir)

    return experiment_dir


def create_experiment_output_dirs(config, model_name: str):
    """Create model-specific folders within one experiment root.

    When EEG_EXPERIMENT_DIR is set (by run_all_models.py), all models share
    that root. Standalone model runs create a new date-first experiment root.
    """
    shared_root = os.environ.get(EXPERIMENT_DIR_ENV)
    if shared_root:
        experiment_dir = Path(shared_root).resolve()
        if not experiment_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment directory from {EXPERIMENT_DIR_ENV} does not exist: "
                f"{experiment_dir}"
            )
        for directory_name in ("models", "results", "plots", "analysis", "experiment_timings", "statistical_tests"):
            (experiment_dir / directory_name).mkdir(exist_ok=True)
    else:
        experiment_dir = create_experiment_root(config).resolve()

    models_dir = experiment_dir / "models" / model_name
    results_dir = experiment_dir / "results" / model_name
    plots_dir = experiment_dir / "plots" / model_name

    for directory in (models_dir, results_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=False)

    return models_dir, results_dir, plots_dir, experiment_dir.name


def print_metric_summary(metrics):
    """
    Prints classification metrics.
    """

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def summarize_metric_list(name, metrics_list):
    """
    Prints mean and standard deviation of metrics across LOSO folds.
    """

    print(f"\n{name}")

    summary = {}

    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        values = np.array([m[metric_name] for m in metrics_list], dtype=float)

        mean = values.mean()
        std = values.std()

        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"] = std

        print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

    if "roc_auc" in metrics_list[0]:
        values = np.array([m["roc_auc"] for m in metrics_list], dtype=float)
        values = values[~np.isnan(values)]

        if len(values) > 0:
            mean = values.mean()
            std = values.std()

            summary["roc_auc_mean"] = mean
            summary["roc_auc_std"] = std

            print(f"roc_auc: {mean:.4f} ± {std:.4f}")

    return summary


def save_fold_metrics_csv(results_path, fold_rows):
    """
    Saves fold-level metrics to CSV.
    """

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "fold",
        "test_subject",
        "test_subjects",
        "val_subjects",
        "best_epoch",
        "best_train_acc",
        "best_val_acc",
        "best_val_loss",
        "epoch_accuracy",
        "epoch_precision",
        "epoch_recall",
        "epoch_f1",
        "epoch_roc_auc",
    ]

    with open(results_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fold_rows)

    print(f"Saved fold metrics to: {results_path}")


def standardize_features_train_val_test(X_train, X_val, X_test):
    """
    Z-score standardization using training data only.

    Works for feature datasets:
        (n_epochs, n_channels, n_features)
        or
        (n_epochs, n_features_flat)

    This avoids leakage from validation/test into training.
    """

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)

    std = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return (
        X_train_std.astype(np.float32),
        X_val_std.astype(np.float32),
        X_test_std.astype(np.float32),
    )


def standardize_raw_train_val_test(X_train, X_val, X_test):
    """
    Per-channel Z-score standardization for raw EEG data.

    Works for raw EEG datasets: (n_epochs, n_channels, n_timepoints)

    Mean and standard deviation are computed using training data only.
    This avoids leakage from validation/test into training.

    For each EEG channel, the mean/std are computed across:
        - all training epochs
        - all timepoints

    Output shape remains unchanged: (n_epochs, n_channels, n_timepoints)
    """

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)

    std = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return (
        X_train_std.astype(np.float32),
        X_val_std.astype(np.float32),
        X_test_std.astype(np.float32),
    )


def save_loso_summary_report(output_path, analysis_name, number_of_folds, train_accuracies, test_accuracies, config, data_augmentation=False, model_config=None):
    """
    Saves LOSO summary report as .txt.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_accuracies = np.asarray(train_accuracies, dtype=float)
    test_accuracies = np.asarray(test_accuracies, dtype=float)

    train_mean = train_accuracies.mean() * 100
    train_std = train_accuracies.std() * 100

    test_mean = test_accuracies.mean() * 100
    test_std = test_accuracies.std() * 100

    train_test_gap = train_mean - test_mean

    training_config = config["training"]
    validation_config = config["validation"]

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("LOSO Cross-Validation Summary\n")
        file.write("=============================\n\n")

        file.write(f"Analysis: {analysis_name}\n")
        file.write(f"Number of folds: {number_of_folds}\n")
        file.write(f"Validation method: {validation_config.get('method', 'LOSO')}\n")
        file.write(f"Validation subjects per fold: {validation_config.get('n_validation_subjects', 1)}\n")
        file.write(f"Data Augmentation: {'ENABLED' if data_augmentation else 'DISABLED'}\n\n")

        file.write("Performance:\n")
        file.write(f"Training Accuracy: {train_mean:.2f}% ± {train_std:.2f}%\n")
        file.write(f"Testing Accuracy: {test_mean:.2f}% ± {test_std:.2f}%\n")
        file.write(f"Train-Test Gap: {train_test_gap:.2f}%\n\n")

        file.write("Training Parameters:\n")
        file.write(f"Epochs: {training_config['epochs']}\n")
        file.write(f"Batch size: {training_config['batch_size']}\n")
        file.write(f"Learning rate: {training_config['learning_rate']}\n")
        file.write(f"Weight decay: {training_config['weight_decay']}\n")
        file.write(f"Patience: {training_config['patience']}\n\n")
        file.write(f"Early stopping metric: {training_config.get('early_stopping_metric', 'val_loss')}\n")

        if model_config is None:
            file.write("Model configuration was not provided.\n")
        else:
            for key, value in model_config.items():
                readable_key = key.replace("_", " ").capitalize()
                file.write(f"{readable_key}: {value}\n")

        file.write("Model Parameters:\n")
        file.write(f"Model: {model_config.get('name', 'EEGNetLike')}\n")
        file.write(f"Dropout: {model_config.get('dropout')}\n")
        file.write(f"Temporal filters: {model_config.get('temporal_filters')}\n")
        file.write(f"Depth multiplier: {model_config.get('depth_multiplier')}\n")
        file.write(f"Temporal kernel size: {model_config.get('temporal_kernel_size')}\n")
        file.write(f"Separable kernel size: {model_config.get('separable_kernel_size')}\n")


import torch


def save_model_checkpoint(model_path, model, model_name: str, model_config: dict, training_config: dict, fold_idx: int, test_subject, val_subject, input_shape: tuple, history: dict, extra: dict | None = None):
    """
    Saves a generic PyTorch model checkpoint.

    Works for:
        - EEGNetLike raw model
        - CNNFeatures1D
        - CNNFeatures2D
        - Hybrid models

    Parameters
    ----------
    input_shape:
        Shape of one input sample excluding batch dimension.

        Examples:
            EEGNetLike: (1, 32, 900)
            CNNFeatures1D: (1, 416)
            CNNFeatures2D: (1, 32, 13)
            Hybrid: can be stored in extra
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "model_config": model_config,
        "training_config": training_config,
        "fold": fold_idx,
        "test_subject": test_subject,
        "val_subject": val_subject,
        "input_shape": input_shape,
        "history": history
    }

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, model_path)

    print(f"Saved model checkpoint: {model_path}")


def load_model_checkpoint(model_path, model_class, model_kwargs: dict, device):
    """
    Loads a generic PyTorch model checkpoint.

    Parameters
    ----------
    model_path: Path to .pt checkpoint.
    model_class:
        Class of the model to instantiate, e.g.
            EEGNetLike
            CNNFeatures1D
            CNNFeatures2D

    model_kwargs: Arguments needed to reconstruct the model.
    device: torch.device

    Returns
    -------
    model: Loaded model in eval mode.
    checkpoint: Full checkpoint dictionary.
    """

    checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Invalid checkpoint format. Expected a dictionary with key 'model_state_dict'."
        )

    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, checkpoint