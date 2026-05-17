import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath("."))

from src.models.cnn_eegnetlike import EEGNetLike
from src.train_eval.split import (create_loso_splits, create_train_val_split_by_subject)
from src.train_eval.train import train_binary_model
from src.train_eval.evaluate import ( predict_binary_model, compute_binary_metrics, aggregate_majority_vote, aggregate_mean_probability, metrics_from_aggregated_results)
from src.train_eval.common import (load_dataset_npz, get_device, create_experiment_output_dirs, print_metric_summary, summarize_metric_list, save_fold_metrics_csv, save_loso_summary_report, save_config_copy, save_model_checkpoint)
from src.train_eval.visualizations import (plot_training_history, plot_training_history_by_iteration, save_confusion_matrix_plot, plot_metric_by_fold, plot_epoch_metrics_summary, plot_roc_curve_for_fold, plot_all_folds_roc_curve,save_final_confusion_matrices, plot_loso_performance_summary)
from src.utils.helpers import load_config


def results_to_y_arrays(results):
    """
    Converts aggregated result dictionaries to y_true and y_pred arrays.
    """

    y_true = np.array([row["true_label"] for row in results], dtype=int)
    y_pred = np.array([row["predicted_label"] for row in results], dtype=int)

    return y_true, y_pred


def main():
    config = load_config("config.yaml")

    if config["validation"].get("method", "LOSO").upper() != "LOSO":
        raise ValueError("This runner currently supports only LOSO validation.")

    raw_dataset_path = config["data"]["raw_dataset_output"]

    if not Path(raw_dataset_path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_dataset_path}\n"
            f"Run first: python src/train_eval/test_dataset_builders.py"
        )

    models_dir, results_dir, plots_dir, experiment_name = create_experiment_output_dirs(config, model_name="raw_eegnetlike")
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_config_copy("config.yaml", results_dir)

    print(f"Experiment name: {experiment_name}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")

    X_raw, y, subjects, conditions = load_dataset_npz(raw_dataset_path)

    print("Loaded raw dataset")
    print("X_raw:", X_raw.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("conditions:", conditions.shape)

    # Raw EEG:
    # from (n_epochs, n_channels, n_timepoints)
    # to (n_epochs, 1, n_channels, n_timepoints)
    X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
    y_tensor_all = torch.tensor(y, dtype=torch.float32)

    print("PyTorch input shape:", X_tensor.shape)

    device = get_device(config)
    print("Device:", device)

    n_channels = X_raw.shape[1]
    n_timepoints = X_raw.shape[2]

    training_config = config["training"]
    model_config = config["model"]["raw_model"]
    outputs_config = config["outputs"]

    validation_seed = config["validation"].get("validation_seed", config["project"]["random_seed"])

    use_validation_subject = config["validation"].get("use_validation_subject", True)

    loso_splits = create_loso_splits(subjects)

    all_epoch_metrics = []
    all_majority_metrics = []
    all_probability_metrics = []

    fold_rows = []
    fold_roc_data = []

    all_y_true_epoch = []
    all_y_pred_epoch = []

    all_y_true_majority = []
    all_y_pred_majority = []

    all_y_true_probability = []
    all_y_pred_probability = []

    train_accuracies_for_summary = []
    test_accuracies_for_summary = []

    for fold_idx, split in enumerate(loso_splits, start=1):
        test_subject = split["test_subject"]

        print("\n" + "=" * 80)
        print(f"Fold {fold_idx}/{len(loso_splits)} | Test subject: {test_subject}")
        print("=" * 80)

        if use_validation_subject:
            train_mask, val_mask, val_subject = create_train_val_split_by_subject(
                subjects=subjects,
                train_mask=split["train_mask"],
                random_seed=validation_seed + fold_idx
            )
        else:
            train_mask = split["train_mask"]
            val_mask = split["test_mask"]
            val_subject = "TEST_USED_AS_VAL"

        test_mask = split["test_mask"]

        print(f"Validation subject: {val_subject}")
        print(f"Train epochs: {train_mask.sum()}")
        print(f"Val epochs: {val_mask.sum()}")
        print(f"Test epochs: {test_mask.sum()}")

        X_train = X_tensor[train_mask]
        y_train = y_tensor_all[train_mask]

        X_val = X_tensor[val_mask]
        y_val = y_tensor_all[val_mask]

        X_test = X_tensor[test_mask]
        y_test = y[test_mask]

        model = EEGNetLike(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            dropout=model_config.get("dropout", 0.5),
            temporal_filters=model_config.get("temporal_filters", 8),
            depth_multiplier=model_config.get("depth_multiplier", 2),
            temporal_kernel_size=model_config.get("temporal_kernel_size", 64),
            separable_kernel_size=model_config.get("separable_kernel_size", 16)
        )

        model, history = train_binary_model(
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
            patience=training_config["patience"]
        )

        # Training progress plots
        plot_training_history(
            history=history,
            output_dir=plots_dir / "training_history",
            fold_idx=fold_idx,
            test_subject=test_subject,
            model_name="Raw EEGNet-like CNN"
        )

        plot_training_history_by_iteration(
            history=history,
            output_dir=plots_dir / "training_history_iteration",
            fold_idx=fold_idx,
            test_subject=test_subject,
            model_name="Raw EEGNet-like CNN"
        )

        y_prob, y_pred = predict_binary_model(
            model=model,
            X=X_test,
            device=device
        )

        epoch_metrics = compute_binary_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob
        )

        print("\nEpoch-level metrics:")
        print_metric_summary(epoch_metrics)

        best_train_acc = history["best_train_acc"]
        if best_train_acc is None:
            best_train_acc = history["train_acc"][-1]

        train_accuracies_for_summary.append(best_train_acc)
        test_accuracies_for_summary.append(epoch_metrics["accuracy"])

        # Save epoch-level confusion matrix per fold
        save_confusion_matrix_plot(
            confusion_matrix=epoch_metrics["confusion_matrix"],
            output_path=plots_dir / "confusion_matrices" / f"epoch_fold_{fold_idx}_subject_{test_subject}.png",
            title=f"Epoch-level confusion matrix - Fold {fold_idx}, Subject {test_subject}"
        )

        # Save ROC per fold
        plot_roc_curve_for_fold(
            y_true=y_test,
            y_prob=y_prob,
            output_path=plots_dir / "roc_curves" / f"roc_fold_{fold_idx}_subject_{test_subject}.png",
            title=f"ROC curve - Fold {fold_idx}, Subject {test_subject}"
        )

        fold_roc_data.append({
            "fold": fold_idx,
            "test_subject": test_subject,
            "y_true": y_test.copy(),
            "y_prob": y_prob.copy()
        })

        all_y_true_epoch.extend(y_test.tolist())
        all_y_pred_epoch.extend(y_pred.tolist())

        test_subjects = subjects[test_mask]
        test_conditions = conditions[test_mask]

        majority_results = aggregate_majority_vote(
            epoch_predictions=y_pred,
            subjects=test_subjects,
            conditions=test_conditions
        )

        majority_metrics = metrics_from_aggregated_results(majority_results)

        print("\nSubject-condition majority vote metrics:")
        print_metric_summary(majority_metrics)

        save_confusion_matrix_plot(
            confusion_matrix=majority_metrics["confusion_matrix"],
            output_path=plots_dir / "confusion_matrices" / f"majority_fold_{fold_idx}_subject_{test_subject}.png",
            title=f"Majority vote confusion matrix - Fold {fold_idx}, Subject {test_subject}"
        )

        y_true_majority, y_pred_majority = results_to_y_arrays(majority_results)
        all_y_true_majority.extend(y_true_majority.tolist())
        all_y_pred_majority.extend(y_pred_majority.tolist())

        probability_results = aggregate_mean_probability(
            epoch_probabilities=y_prob,
            subjects=test_subjects,
            conditions=test_conditions
        )

        probability_metrics = metrics_from_aggregated_results(probability_results)

        print("\nSubject-condition mean probability metrics:")
        print_metric_summary(probability_metrics)

        save_confusion_matrix_plot(
            confusion_matrix=probability_metrics["confusion_matrix"],
            output_path=plots_dir / "confusion_matrices" / f"probability_fold_{fold_idx}_subject_{test_subject}.png",
            title=f"Mean probability confusion matrix - Fold {fold_idx}, Subject {test_subject}",
        )

        y_true_probability, y_pred_probability = results_to_y_arrays(probability_results)
        all_y_true_probability.extend(y_true_probability.tolist())
        all_y_pred_probability.extend(y_pred_probability.tolist())

        all_epoch_metrics.append(epoch_metrics)
        all_majority_metrics.append(majority_metrics)
        all_probability_metrics.append(probability_metrics)

        fold_rows.append({
            "fold": fold_idx,
            "test_subject": test_subject,
            "val_subject": val_subject,
            "epoch_accuracy": epoch_metrics["accuracy"],
            "epoch_precision": epoch_metrics["precision"],
            "epoch_recall": epoch_metrics["recall"],
            "epoch_f1": epoch_metrics["f1"],
            "epoch_roc_auc": epoch_metrics.get("roc_auc", np.nan),
            "majority_accuracy": majority_metrics["accuracy"],
            "majority_precision": majority_metrics["precision"],
            "majority_recall": majority_metrics["recall"],
            "majority_f1": majority_metrics["f1"],
            "probability_accuracy": probability_metrics["accuracy"],
            "probability_precision": probability_metrics["precision"],
            "probability_recall": probability_metrics["recall"],
            "probability_f1": probability_metrics["f1"]
        })

        if outputs_config.get("save_models", True):
            model_path = models_dir / f"raw_eegnetlike_fold_{fold_idx}_subject_{test_subject}.pt"

            save_model_checkpoint(
                model_path=model_path,
                model=model,
                model_name="EEGNetLike",
                model_config=model_config,
                training_config=training_config,
                fold_idx=fold_idx,
                test_subject=test_subject,
                val_subject=val_subject,
                input_shape=(1, n_channels, n_timepoints),
                history=history,
                extra={
                    "n_channels": n_channels,
                    "n_timepoints": n_timepoints,
                }
            )

    print("\n" + "=" * 80)
    print("FINAL LOSO SUMMARY - RAW EEGNET-LIKE CNN")
    print("=" * 80)

    summarize_metric_list("Epoch-level", all_epoch_metrics)
    summarize_metric_list("Majority vote", all_majority_metrics)
    summarize_metric_list("Mean probability", all_probability_metrics)

    if outputs_config.get("save_metrics", True):
        save_fold_metrics_csv(
            results_path=results_dir / "raw_eegnetlike_loso_metrics.csv",
            fold_rows=fold_rows
        )

    plot_loso_performance_summary(
        train_accuracies=train_accuracies_for_summary,
        test_accuracies=test_accuracies_for_summary,
        output_path=plots_dir / "summary" / "loso_performance_summary.png",
        title="LOSO cross-validation: Raw EEGNet-like CNN",
        chance_level=0.5
    )

    save_loso_summary_report(
        output_path=results_dir / "raw_eegnetlike_loso_summary.txt",
        analysis_name="Raw EEGNet-like CNN: Before vs. After",
        number_of_folds=len(loso_splits),
        train_accuracies=train_accuracies_for_summary,
        test_accuracies=test_accuracies_for_summary,
        config=config,
        data_augmentation=False
    )

    # Final pooled confusion matrices
    save_final_confusion_matrices(
        all_y_true_epoch=all_y_true_epoch,
        all_y_pred_epoch=all_y_pred_epoch,
        all_y_true_majority=all_y_true_majority,
        all_y_pred_majority=all_y_pred_majority,
        all_y_true_probability=all_y_true_probability,
        all_y_pred_probability=all_y_pred_probability,
        output_dir=plots_dir / "final_confusion_matrices",
    )

    # Fold-level metric plots
    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_accuracy",
        output_path=plots_dir / "metrics_by_fold" / "epoch_accuracy_by_fold.png",
        title="Epoch-level accuracy by LOSO fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_f1",
        output_path=plots_dir / "metrics_by_fold" / "epoch_f1_by_fold.png",
        title="Epoch-level F1 by LOSO fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="epoch_roc_auc",
        output_path=plots_dir / "metrics_by_fold" / "epoch_roc_auc_by_fold.png",
        title="Epoch-level ROC-AUC by LOSO fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="majority_accuracy",
        output_path=plots_dir / "metrics_by_fold" / "majority_accuracy_by_fold.png",
        title="Subject-condition majority vote accuracy by LOSO fold",
    )

    plot_metric_by_fold(
        fold_rows=fold_rows,
        metric_name="probability_accuracy",
        output_path=plots_dir / "metrics_by_fold" / "probability_accuracy_by_fold.png",
        title="Subject-condition mean probability accuracy by LOSO fold",
    )

    plot_epoch_metrics_summary(
        fold_rows=fold_rows,
        output_path=plots_dir / "metrics_by_fold" / "epoch_metrics_summary.png",
    )

    plot_all_folds_roc_curve(
        fold_roc_data=fold_roc_data,
        output_path=plots_dir / "roc_curves" / "roc_all_folds.png",
    )

    print(f"\nPlots saved to: {plots_dir}")


if __name__ == "__main__":
    main()