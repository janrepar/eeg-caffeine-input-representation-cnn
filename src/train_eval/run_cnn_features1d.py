import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath("."))

from src.models.cnn_features_1d import CNNFeatures1D
from src.train_eval.split import (create_loso_splits, create_groupkfold_splits, create_train_val_split_by_subject,
)
from src.train_eval.train import train_model
from src.train_eval.evaluate import (predict_multiclass_model, compute_binary_metrics)
from src.train_eval.common import (load_dataset_npz, get_device, create_experiment_output_dirs, print_metric_summary, summarize_metric_list, save_fold_metrics_csv, save_loso_summary_report, save_config_copy, save_model_checkpoint,standardize_features_train_val_test, set_random_seed, print_model_parameter_count)
from src.train_eval.visualizations import ( plot_training_history, plot_training_history_by_iteration, save_confusion_matrix_plot, plot_metric_by_fold, plot_epoch_metrics_summary, plot_roc_curve_for_fold, plot_all_folds_roc_curve, save_final_confusion_matrix, plot_loso_performance_summary,)
from src.utils.helpers import load_config


def main():
    config = load_config("config.yaml")
    set_random_seed(config["project"]["random_seed"])

    run_start_time = time.perf_counter()

    validation_method = config["validation"].get("method", "LOSO").upper()

    if validation_method not in ["LOSO", "GROUPKFOLD"]:
        raise ValueError(f"Unsupported validation method: {validation_method}")

    feature_dataset_path = config["data"]["feature_dataset_output"]

    if not Path(feature_dataset_path).exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {feature_dataset_path}\n"
            f"Run first: python src/train_eval/test_dataset_builders.py"
        )

    models_dir, results_dir, plots_dir, experiment_name = create_experiment_output_dirs(
        config,
        model_name="features1d_cnn",
    )
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config_copy("config.yaml", results_dir)

    print(f"Experiment name: {experiment_name}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")

    X_feat, y, subjects, conditions, groups = load_dataset_npz(feature_dataset_path)

    print("Loaded feature dataset")
    print("X_feat:", X_feat.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("conditions:", conditions.shape)

    if groups is not None:
        print("groups:", groups.shape)

    analysis_type = config.get("experiment", {}).get("analysis_type", "before_vs_after")

    print(f"\nAnalysis type: {analysis_type}")
    print(f"\nValidation method: {validation_method}")

    if analysis_type == "caffeine_before_vs_after":
        if groups is None:
            raise ValueError("Dataset does not contain groups. Rebuild dataset with groups.")

        mask = groups == "Caffeine"

        X_feat = X_feat[mask]
        y = y[mask]
        subjects = subjects[mask]
        conditions = conditions[mask]
        groups = groups[mask]

        print("\nUsing only Caffeine group.")
        print("X_feat:", X_feat.shape)
        print("Subjects:", np.unique(subjects))
        print("Number of subjects:", len(np.unique(subjects)))

    elif analysis_type == "placebo_before_vs_after":
        if groups is None:
            raise ValueError("Dataset does not contain groups. Rebuild dataset with groups.")

        mask = groups == "Placebo"

        X_feat = X_feat[mask]
        y = y[mask]
        subjects = subjects[mask]
        conditions = conditions[mask]
        groups = groups[mask]

        print("\nUsing only Placebo group.")
        print("X_feat:", X_feat.shape)
        print("Subjects:", np.unique(subjects))
        print("Number of subjects:", len(np.unique(subjects)))

    elif analysis_type == "before_vs_after":
        print("\nUsing all subjects: Before vs After.")

    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    print("\nDataset after analysis filtering:")
    print("X_feat:", X_feat.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("conditions:", conditions.shape)
    print("Unique subjects:", np.unique(subjects))
    print("Number of subjects:", len(np.unique(subjects)))
    print("Before epochs:", np.sum(y == 0))
    print("After epochs:", np.sum(y == 1))

    print("Feature input shape before fold-wise standardization:", X_feat.shape)

    device = get_device(config)
    print("Device:", device)

    n_channels = X_feat.shape[1]
    n_features = X_feat.shape[2]

    training_config = config["training"]
    model_config = config["model"]["features_1d_model"]
    outputs_config = config["outputs"]

    validation_seed = config["validation"].get(
        "validation_seed",
        config["project"]["random_seed"],
    )

    use_validation_subject = config["validation"].get("use_validation_subject", True)
    n_validation_subjects = config["validation"].get("n_validation_subjects", 1)

    early_stopping_metric = training_config.get("early_stopping_metric", "val_loss")
    label_smoothing = training_config.get("label_smoothing", 0.0)

    print(f"Validation subjects per fold: {n_validation_subjects}")
    print(f"Early stopping metric: {early_stopping_metric}")
    print(f"Label smoothing: {label_smoothing}")

    if validation_method == "LOSO":
        splits = create_loso_splits(subjects)

    elif validation_method == "GROUPKFOLD":
        n_splits = config["validation"].get("n_splits", 5)

        splits = create_groupkfold_splits(
            subjects=subjects,
            n_splits=n_splits,
        )

    else:
        raise ValueError(f"Unsupported validation method: {validation_method}")

    all_epoch_metrics = []

    fold_rows = []
    fold_roc_data = []

    all_y_true_epoch = []
    all_y_pred_epoch = []

    train_accuracies_for_summary = []
    test_accuracies_for_summary = []

    for fold_idx, split in enumerate(splits, start=1):
        if "test_subjects" in split:
            test_subjects = np.asarray(split["test_subjects"])
        else:
            test_subjects = np.asarray([split["test_subject"]])

        test_subject_label = "_".join(map(str, test_subjects))

        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{len(splits)} | Test subjects: {test_subjects}")
        print("=" * 80)

        if use_validation_subject:
            train_mask, val_mask, val_subjects = create_train_val_split_by_subject(
                subjects=subjects,
                train_mask=split["train_mask"],
                random_seed=validation_seed + fold_idx,
                n_validation_subjects=n_validation_subjects,
            )
        else:
            train_mask = split["train_mask"]
            val_mask = split["test_mask"]
            val_subjects = np.array(["TEST_USED_AS_VAL"])

        test_mask = split["test_mask"]

        print(f"Validation subjects: {val_subjects}")
        print(f"Train epochs: {train_mask.sum()}")
        print(f"Val epochs: {val_mask.sum()}")
        print(f"Test epochs: {test_mask.sum()}")

        X_train_np = X_feat[train_mask]
        y_train_np = y[train_mask]

        X_val_np = X_feat[val_mask]
        y_val_np = y[val_mask]

        X_test_np = X_feat[test_mask]
        y_test = y[test_mask]

        X_train_np, X_val_np, X_test_np = standardize_features_train_val_test(
            X_train_np,
            X_val_np,
            X_test_np,
        )

        # IMPORTANT:
        # Conv1d expects input shape:
        #     (batch_size, n_channels, n_features)
        # Therefore do NOT use unsqueeze(1) here.
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)

        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.long)

        X_test = torch.tensor(X_test_np, dtype=torch.float32)

        print("X_train:", X_train.shape)
        print("X_val:", X_val.shape)
        print("X_test:", X_test.shape)

        if validation_method == "LOSO":
            hpo_path = Path(config["outputs"]["output_dir"]) / "hyperparameter_optimization" / "features1d" / f"best_params_subject_{test_subject_label}.json"
            if not hpo_path.is_file():
                raise FileNotFoundError(f"Missing Optuna parameters for test subject {test_subject_label}: {hpo_path}")
            with open(hpo_path, encoding="utf-8") as hpo_file:
                optimized_params = json.load(hpo_file)["best_params"]
            model_config.update({key: value for key, value in optimized_params.items() if key not in {"learning_rate", "weight_decay"}})
            training_config.update({key: value for key, value in optimized_params.items() if key in {"learning_rate", "weight_decay", "label_smoothing"}})
            label_smoothing = training_config.get("label_smoothing", 0.0)
        model = CNNFeatures1D(
            n_channels=n_channels,
            n_features=n_features,
            n_classes=2,
            dropout=model_config.get("dropout", 0.5),
            conv1_filters=model_config.get("conv1_filters", 32),
            conv2_filters=model_config.get("conv2_filters", 64),
            kernel_size_1=model_config.get("kernel_size_1", 3),
            kernel_size_2=model_config.get("kernel_size_2", 3),
            hidden_units=model_config.get("hidden_units", 64),
        )

        print_model_parameter_count(model, "Features1D CNN")

        model, history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            patience=training_config["patience"],
            validation_frequency=training_config.get("validation_frequency", 30),
            early_stopping_metric=early_stopping_metric,
            label_smoothing=label_smoothing,
        )

        print(
            f"Best epoch: {history['best_epoch']} | "
            f"Best val loss: {history['best_val_loss']:.4f} | "
            f"Best val acc: {history['best_val_acc']:.4f} | "
            f"Best train acc: {history['best_train_acc']:.4f}"
        )

        plot_training_history(
            history=history,
            output_dir=plots_dir / "training_history",
            fold_idx=fold_idx,
            test_subject=test_subject_label,
            model_name="Features1D CNN",
        )

        plot_training_history_by_iteration(
            history=history,
            output_dir=plots_dir / "training_history_iteration",
            fold_idx=fold_idx,
            test_subject=test_subject_label,
            model_name="Features1D CNN",
        )

        y_prob, y_pred = predict_multiclass_model(
            model=model,
            X=X_test,
            device=device,
        )

        epoch_metrics = compute_binary_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
        )

        print("\nEpoch-level metrics:")
        print_metric_summary(epoch_metrics)

        best_train_acc = history["best_train_acc"]
        if best_train_acc is None:
            best_train_acc = history["train_acc"][-1]

        train_accuracies_for_summary.append(best_train_acc)
        test_accuracies_for_summary.append(epoch_metrics["accuracy"])

        save_confusion_matrix_plot(
            confusion_matrix=epoch_metrics["confusion_matrix"],
            output_path=plots_dir / "confusion_matrices" / f"epoch_fold_{fold_idx}_subjects_{test_subject_label}.png",
            title=f"Epoch-level confusion matrix - Fold {fold_idx}, Subjects {test_subject_label}"
        )

        plot_roc_curve_for_fold(
            y_true=y_test,
            y_prob=y_prob,
            output_path=plots_dir / "roc_curves" / f"roc_fold_{fold_idx}_subjects_{test_subject_label}.png",
            title=f"ROC curve - Fold {fold_idx}, Subjects {test_subject_label}"
        )

        fold_roc_data.append({
            "fold": fold_idx,
            "test_subject": test_subject_label,
            "test_subjects": ",".join(map(str, test_subjects)),
            "y_true": y_test.copy(),
            "y_prob": y_prob.copy()
        })

        all_y_true_epoch.extend(y_test.tolist())
        all_y_pred_epoch.extend(y_pred.tolist())

        all_epoch_metrics.append(epoch_metrics)

        fold_rows.append({
            "fold": fold_idx,
            "test_subject": test_subject_label,
            "test_subjects": ",".join(map(str, test_subjects)),
            "val_subjects": ",".join(map(str, val_subjects)),
            "best_epoch": history["best_epoch"],
            "best_train_acc": history["best_train_acc"],
            "best_val_acc": history["best_val_acc"],
            "best_val_loss": history["best_val_loss"],
            "epoch_accuracy": epoch_metrics["accuracy"],
            "epoch_precision": epoch_metrics["precision"],
            "epoch_recall": epoch_metrics["recall"],
            "epoch_f1": epoch_metrics["f1"],
            "epoch_roc_auc": epoch_metrics.get("roc_auc", np.nan)
        })

        if outputs_config.get("save_models", True):
            model_path = models_dir / f"features1d_cnn_fold_{fold_idx}_subjects_{test_subject_label}.pt"

            save_model_checkpoint(
                model_path=model_path,
                model=model,
                model_name="CNNFeatures1D",
                model_config=model_config,
                training_config=training_config,
                fold_idx=fold_idx,
                test_subject=test_subject_label,
                val_subject=",".join(map(str, val_subjects)),
                input_shape=(n_channels, n_features),
                history=history,
                extra={
                    "n_channels": n_channels,
                    "n_features": n_features,
                    "n_classes": 2,
                    "test_subjects": ",".join(map(str, test_subjects)),
                    "validation_method": validation_method
                },
            )

    print("\n" + "=" * 80)
    print(f"FINAL {validation_method} SUMMARY - FEATURES1D CNN")
    print("=" * 80)

    summarize_metric_list("Epoch-level", all_epoch_metrics)

    if outputs_config.get("save_metrics", True):
        save_fold_metrics_csv(
            results_path=results_dir / f"features1d_cnn_{validation_method.lower()}_metrics.csv",
            fold_rows=fold_rows,
        )

    plot_loso_performance_summary(
        train_accuracies=train_accuracies_for_summary,
        test_accuracies=test_accuracies_for_summary,
        output_path=plots_dir / "summary" / f"{validation_method.lower()}_performance_summary.png",
        title=f"{validation_method} cross-validation: Features1D CNN",
        chance_level=0.5,
    )

    save_loso_summary_report(
        output_path=results_dir / f"features1d_cnn_{validation_method.lower()}_summary.txt",
        analysis_name=f"Features1D CNN: {analysis_type}",
        number_of_folds=len(splits),
        train_accuracies=train_accuracies_for_summary,
        test_accuracies=test_accuracies_for_summary,
        config=config,
        data_augmentation=False,
        model_config=model_config
    )

    save_final_confusion_matrix(
        all_y_true_epoch=all_y_true_epoch,
        all_y_pred_epoch=all_y_pred_epoch,
        output_dir=plots_dir / "final_confusion_matrix",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_accuracy",
        output_path=plots_dir / "metrics_by_fold" / "epoch_accuracy_by_fold.png",
        title=f"Epoch-level accuracy by {validation_method} fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_f1",
        output_path=plots_dir / "metrics_by_fold" / "epoch_f1_by_fold.png",
        title=f"Epoch-level F1 by {validation_method} fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_roc_auc",
        output_path=plots_dir / "metrics_by_fold" / "epoch_roc_auc_by_fold.png",
        title=f"Epoch-level ROC-AUC by {validation_method} fold",
    )

    plot_epoch_metrics_summary(
        fold_rows=fold_rows,
        output_path=plots_dir / "metrics_by_fold" / "epoch_metrics_summary.png",
    )

    plot_all_folds_roc_curve(
        fold_roc_data=fold_roc_data,
        output_path=plots_dir / "roc_curves" / "roc_all_folds.png",
    )

    run_end_time = time.perf_counter()
    run_duration_seconds = run_end_time - run_start_time
    run_duration_minutes = run_duration_seconds / 60

    print(f"\nTotal runtime: {run_duration_minutes:.2f} minutes")

    with open(results_dir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"runtime_seconds: {run_duration_seconds:.2f}\n")
        f.write(f"runtime_minutes: {run_duration_minutes:.2f}\n")

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()