from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.bipartite.basic import color
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc


def plot_training_history(history: dict, output_dir, fold_idx: int, test_subject: str, model_name: str):
    """
    Saves training curves as a training-progress plot.

    Creates:
        - loss curve
        - accuracy curve
        - combined loss/accuracy figure
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # 1. Loss only
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Training loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss - Fold {fold_idx}, Subject {test_subject}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"loss_fold_{fold_idx}_subject_{test_subject}.png", dpi=300)
    plt.close()

    # 2. Accuracy only
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Training accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"{model_name} - Accuracy - Fold {fold_idx}, Subject {test_subject}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"accuracy_fold_{fold_idx}_subject_{test_subject}.png", dpi=300)
    plt.close()

    # 3. Combined figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(epochs, history["train_acc"], label="Training accuracy")
    axes[0].plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        f"{model_name} Training Progress - Fold {fold_idx}, Subject {test_subject}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], label="Training loss")
    axes[1].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"training_progress_fold_{fold_idx}_subject_{test_subject}.png", dpi=300)
    plt.close()


def plot_training_history_by_iteration(history: dict, output_dir, fold_idx: int, test_subject: str, model_name: str):
    """
    Saves training progress plot by iteration.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(
        history["iter"],
        history["iter_train_acc"],
        label="Training accuracy"
    )

    axes[0].plot(
        history["iter_val_iter"],
        history["iter_val_acc"],
        marker="o",
        linestyle="-",
        label="Validation accuracy"
    )

    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        f"{model_name} Training Progress by Iteration - Fold {fold_idx}, Subject {test_subject}"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history["iter"],
        history["iter_train_loss"],
        label="Training loss"
    )

    axes[1].plot(
        history["iter_val_iter"],
        history["iter_val_loss"],
        marker="o",
        linestyle="-",
        label="Validation loss"
    )

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"training_progress_iteration_fold_{fold_idx}_subject_{test_subject}.png",
        dpi=300
    )
    plt.close()


def save_confusion_matrix_plot(confusion_matrix, output_path, title: str,):
    """
    Saves confusion matrix plot.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["Before", "After"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(values_format="d", ax=ax)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_by_fold(fold_rows: list[dict], metric_name: str, output_path, title: str, chance_level: float = 0.5,):
    """
    Saves bar plot for one metric across LOSO folds.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    folds = [row["fold"] for row in fold_rows]
    values = [row[metric_name] for row in fold_rows]

    plt.figure(figsize=(12, 5))
    plt.bar(folds, values)
    plt.axhline(chance_level, linestyle="--", color="red", linewidth=1, label="Chance level")
    plt.xlabel("LOSO fold")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_epoch_metrics_summary(fold_rows: list[dict], output_path):
    """
    Saves combined plot with accuracy, F1 and ROC-AUC across folds.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    folds = np.array([row["fold"] for row in fold_rows])

    accuracy = np.array([row["epoch_accuracy"] for row in fold_rows])
    f1 = np.array([row["epoch_f1"] for row in fold_rows])
    roc_auc = np.array([row["epoch_roc_auc"] for row in fold_rows])

    plt.figure(figsize=(12, 6))
    plt.plot(folds, accuracy, marker="o", label="Accuracy")
    plt.plot(folds, f1, marker="o", label="F1")
    plt.plot(folds, roc_auc, marker="o", label="ROC-AUC")
    plt.axhline(0.5, linestyle="--", linewidth=1, label="Chance level")

    plt.xlabel("LOSO fold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.title("Epoch-level metrics across LOSO folds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curve_for_fold(y_true, y_prob, output_path, title: str):
    """
    Saves ROC curve for one fold.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance level")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_folds_roc_curve(fold_roc_data: list[dict], output_path):
    """
    Saves ROC curves for all folds in one figure.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))

    auc_values = []

    for row in fold_roc_data:
        y_true = np.asarray(row["y_true"]).astype(int)
        y_prob = np.asarray(row["y_prob"]).astype(float)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)

        plt.plot(
            fpr,
            tpr,
            alpha=0.35,
            label=f"Fold {row['fold']} AUC={roc_auc:.2f}"
        )

    mean_auc = np.mean(auc_values)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance level")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves across LOSO folds (mean AUC = {mean_auc:.3f})")
    plt.legend(fontsize=7, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_final_confusion_matrices(all_y_true_epoch, all_y_pred_epoch, all_y_true_majority, all_y_pred_majority, all_y_true_probability, all_y_pred_probability, output_dir):
    """
    Saves final confusion matrices pooled across all folds.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_epoch = confusion_matrix(all_y_true_epoch, all_y_pred_epoch, labels=[0, 1])
    cm_majority = confusion_matrix(all_y_true_majority, all_y_pred_majority, labels=[0, 1])
    cm_probability = confusion_matrix(all_y_true_probability, all_y_pred_probability, labels=[0, 1])

    save_confusion_matrix_plot(
        cm_epoch,
        output_dir / "final_confusion_matrix_epoch_level.png",
        "Final epoch-level confusion matrix"
    )

    save_confusion_matrix_plot(
        cm_majority,
        output_dir / "final_confusion_matrix_majority_vote.png",
        "Final subject-condition majority vote confusion matrix"
    )

    save_confusion_matrix_plot(
        cm_probability,
        output_dir / "final_confusion_matrix_mean_probability.png",
        "Final subject-condition mean probability confusion matrix"
    )


def plot_loso_performance_summary(train_accuracies, test_accuracies, output_path, title="LOSO cross-validation performance summary", chance_level=0.5):
    """
    Saves LOSO performance summary plot.

    train_accuracies and test_accuracies should be in range [0, 1].
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_accuracies = np.asarray(train_accuracies, dtype=float)
    test_accuracies = np.asarray(test_accuracies, dtype=float)

    means = np.array([
        train_accuracies.mean(),
        test_accuracies.mean(),
    ])

    stds = np.array([
        train_accuracies.std(),
        test_accuracies.std(),
    ])

    labels = ["Training", "Testing"]

    plt.figure(figsize=(9, 6))

    bars = plt.bar(labels, means * 100, yerr=stds * 100, capsize=8)

    plt.axhline(
        chance_level * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Chance level"
    )

    plt.text(
        1.55,
        chance_level * 100,
        "Chance Level",
        va="center",
        fontsize=11
    )

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{mean * 100:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()