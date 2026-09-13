"""Create thesis-ready learning-curve and confusion-matrix summaries.

The training checkpoints contain the complete train/validation history of each
outer-validation fold.  This script aggregates those histories into a single
two-panel (accuracy and loss) figure for every validation protocol.  It also
creates compact contact sheets of all fold-level learning curves and confusion
matrices for use in an appendix.

Example
-------
python src/train_eval/aggregate_experiment_plots.py \
  --groupkfold-experiment outputs/experiments/2026-09-01_212658_GROUPKFOLD_nvalsubj2 \
  --loso-experiment outputs/experiments/2026-08-01_104659_LOSO_nvalsubj2
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.utils.helpers import load_config


MODEL_ORDER = ("raw_eegnetlike", "features1d_cnn", "features2d_cnn", "hybrid_cnn")
CONFUSION_MATRIX_FIELDS = (
    "cm_true_before_pred_before",
    "cm_true_before_pred_after",
    "cm_true_after_pred_before",
    "cm_true_after_pred_after",
)
MODEL_LABELS = {
    "raw_eegnetlike": "Raw EEGNet-like",
    "features1d_cnn": "1D feature CNN",
    "features2d_cnn": "2D feature CNN",
    "hybrid_cnn": "Hybrid CNN",
}
MODEL_LABELS_SL = {
    "raw_eegnetlike": "EEGNet-like CNN z očiščenimi podatki EEG",
    "features1d_cnn": "1D CNN z značilkami",
    "features2d_cnn": "2D CNN z značilkami",
    "hybrid_cnn": "Hibridni CNN",
}
PROTOCOL_LABELS_SL = {
    "Leave-one-subject-out": "LOSO",
    "GroupKFold": "GroupKFold",
}
PROTOCOLS = (
    ("LOSO", "Leave-one-subject-out", "loso"),
    ("GroupKFold", "GroupKFold", "groupkfold"),
)
FOLD_NUMBER = re.compile(r"fold_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groupkfold-experiment",
        type=Path,
        default=Path("outputs/experiments/2026-09-01_212658_GROUPKFOLD_nvalsubj2"),
        help="Experiment directory produced with GroupKFold.",
    )
    parser.add_argument(
        "--loso-experiment",
        type=Path,
        default=Path("outputs/experiments/2026-08-01_104659_LOSO_nvalsubj2"),
        help="Experiment directory produced with LOSO.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/thesis_figures/model_diagnostics"),
        help="Directory in which summary figures will be written.",
    )
    return parser.parse_args()


def sort_by_fold(paths: Iterable[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = FOLD_NUMBER.search(path.name)
        return (int(match.group(1)) if match else math.inf, path.name)

    return sorted(paths, key=key)


def model_checkpoints(experiment_dir: Path, model_key: str) -> list[Path]:
    paths = sort_by_fold(
        (experiment_dir / "models" / model_key).glob(f"{model_key}_fold_*.pt")
    )
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints for {model_key!r} found in {experiment_dir / 'models'}"
        )
    return paths


def load_histories(experiment_dir: Path, model_key: str) -> list[dict[str, np.ndarray]]:
    histories: list[dict[str, np.ndarray]] = []
    required = ("train_acc", "val_acc", "train_loss", "val_loss")
    for checkpoint_path in model_checkpoints(experiment_dir, model_key):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = checkpoint.get("history", {})
        missing = [key for key in required if key not in history]
        if missing:
            raise ValueError(f"{checkpoint_path} has no history values for: {missing}")
        histories.append({key: np.asarray(history[key], dtype=float) for key in required})
    return histories


def mean_and_sd(histories: list[dict[str, np.ndarray]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    # Training can stop at different epochs across folds.  Restricting the
    # aggregate to their common horizon keeps the number of contributing folds
    # constant at every displayed epoch.
    common_epochs = min(len(history[metric]) for history in histories)
    values = np.stack([history[metric][:common_epochs] for history in histories])
    return values.mean(axis=0), values.std(axis=0)


def save_mean_learning_curves(
    experiment_dir: Path, protocol_title: str, protocol_key: str, output_dir: Path
) -> Path:
    """Plot equal-fold-weighted mean +/- SD curves for all four model types."""
    all_histories = {model: load_histories(experiment_dir, model) for model in MODEL_ORDER}
    fold_count = len(next(iter(all_histories.values())))
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 7.2), sharex=False)

    for column, model_key in enumerate(MODEL_ORDER):
        histories = all_histories[model_key]
        for row, (train_key, validation_key, ylabel) in enumerate(
            (("train_acc", "val_acc", "Accuracy"), ("train_loss", "val_loss", "Loss"))
        ):
            axis = axes[row, column]
            for metric, label, color in (
                (train_key, "Training", "#1f77b4"),
                (validation_key, "Validation", "#d62728"),
            ):
                mean, sd = mean_and_sd(histories, metric)
                epochs = np.arange(1, len(mean) + 1)
                axis.plot(epochs, mean, color=color, linewidth=2, label=label)
                axis.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.16)

            if row == 0:
                axis.set_title(MODEL_LABELS[model_key], fontweight="bold")
                axis.set_ylim(0, 1)
            axis.set_xlabel("Epoch")
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            if row == 0 and column == 0:
                axis.legend(loc="lower right")

    fig.suptitle(
        f"{protocol_title}: mean training and validation curves across outer folds (n = {fold_count})\n"
        "Lines show the fold mean; shaded bands show ±1 standard deviation.",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    output_path = output_dir / "main_text" / f"mean_learning_curves_{protocol_key}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def get_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    offset = ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2)
    canvas.paste(copy, offset)
    return canvas


def save_contact_sheet(
    image_paths: list[Path],
    title: str,
    output_path: Path,
    columns: int = 3,
    labels: list[str] | None = None,
    tile_size: tuple[int, int] = (640, 430),
) -> Path:
    """Make a legible contact sheet while retaining the original plot labels."""
    if not image_paths:
        raise FileNotFoundError(f"No images available for contact sheet: {title}")
    if labels is not None and len(labels) != len(image_paths):
        raise ValueError("When supplied, labels must match the number of images.")

    scale = tile_size[0] / 640
    header_height = round(95 * scale)
    label_height = round(34 * scale)
    margin = round(20 * scale)
    rows = math.ceil(len(image_paths) / columns)
    width = 2 * margin + columns * tile_size[0]
    height = header_height + margin + rows * (tile_size[1] + label_height + margin)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(round(32 * scale))
    label_font = get_font(round(20 * scale))
    draw.text((margin, round(25 * scale)), title, fill="black", font=title_font)

    for index, image_path in enumerate(image_paths):
        row, column = divmod(index, columns)
        x = margin + column * tile_size[0]
        y = header_height + margin + row * (tile_size[1] + label_height + margin)
        with Image.open(image_path) as image:
            sheet.paste(fit_image(image, tile_size), (x, y))
        label = (
            labels[index]
            if labels is not None
            else image_path.stem.replace("training_progress_", "").replace("epoch_", "")
        )
        draw.text((x, y + tile_size[1] + 5), label, fill="black", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def load_fold_metrics(experiment_dir: Path, model_key: str) -> list[dict[str, str]]:
    """Load the per-fold epoch metrics written by a completed experiment."""
    paths = list((experiment_dir / "results" / model_key).glob(f"{model_key}_*_metrics.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one metrics CSV for {model_key!r}; found {paths}")
    with paths[0].open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: int(row["fold"]))


def load_subject_epoch_counts(experiment_dir: Path) -> dict[str, tuple[int, int]]:
    """Fallback for legacy metrics that do not persist confusion-matrix cells."""
    config = load_config(experiment_dir / "config_used.yaml")
    dataset_path = Path(config["data"]["raw_dataset_output"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing processed raw dataset: {dataset_path}")
    with np.load(dataset_path, allow_pickle=True) as data:
        subjects = data["subjects"].astype(str)
        labels = data["y"].astype(int)
        groups = data["groups"].astype(str)

    analysis_type = config.get("experiment", {}).get("analysis_type", "before_vs_after")
    if analysis_type == "caffeine_before_vs_after":
        analysis_mask = np.char.lower(groups) == "caffeine"
    elif analysis_type == "placebo_before_vs_after":
        analysis_mask = np.char.lower(groups) == "placebo"
    elif analysis_type == "before_vs_after":
        analysis_mask = np.ones(len(labels), dtype=bool)
    else:
        raise ValueError(f"Unknown analysis type in {experiment_dir}: {analysis_type}")

    counts: dict[str, tuple[int, int]] = {}
    for subject in np.unique(subjects[analysis_mask]):
        subject_mask = analysis_mask & (subjects == subject)
        counts[subject] = (
            int(np.count_nonzero(labels[subject_mask] == 0)),
            int(np.count_nonzero(labels[subject_mask] == 1)),
        )
    return counts


def infer_confusion_matrix(
    metrics: dict[str, str], subject_epoch_counts: dict[str, tuple[int, int]]
) -> np.ndarray:
    """Recover exact integer confusion-matrix counts from metrics and test labels."""
    if all(metrics.get(field) not in (None, "") for field in CONFUSION_MATRIX_FIELDS):
        values = [int(metrics[field]) for field in CONFUSION_MATRIX_FIELDS]
        return np.asarray(values, dtype=int).reshape(2, 2)

    accuracy = Fraction(metrics["epoch_accuracy"]).limit_denominator(100_000)
    precision = Fraction(metrics["epoch_precision"]).limit_denominator(100_000)
    recall = Fraction(metrics["epoch_recall"]).limit_denominator(100_000)
    subject_ids = (metrics.get("test_subjects") or metrics["test_subject"]).split(",")
    try:
        negatives, positives = map(sum, zip(*(subject_epoch_counts[subject] for subject in subject_ids)))
    except KeyError as error:
        raise KeyError(f"Test participant is missing from the caffeine dataset: {error}") from error

    if recall == 0:
        true_positive = 0
    else:
        true_positive_fraction = recall * positives
        if true_positive_fraction.denominator != 1:
            raise ValueError(f"Recall does not match test labels: {metrics}")
        true_positive = int(true_positive_fraction)
    false_negative = positives - true_positive

    if precision == 0:
        true_negative_fraction = accuracy * (negatives + positives)
        if true_negative_fraction.denominator != 1:
            raise ValueError(f"Accuracy does not match test labels: {metrics}")
        true_negative = int(true_negative_fraction)
        false_positive = negatives - true_negative
    else:
        false_positive_fraction = true_positive * (1 - precision) / precision
        if false_positive_fraction.denominator != 1:
            raise ValueError(f"Precision does not match test labels: {metrics}")
        false_positive = int(false_positive_fraction)
        true_negative = negatives - false_positive

    matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    total = matrix.sum()
    if Fraction(int(true_negative + true_positive), int(total)) != accuracy:
        raise ValueError(f"Accuracy does not match reconstructed matrix: {metrics}")
    return matrix


def save_large_text_confusion_sheet(
    experiment_dir: Path,
    model_key: str,
    protocol_title: str,
    output_path: Path,
) -> Path:
    """Create the original three-column layout with thesis-readable labels."""
    rows = load_fold_metrics(experiment_dir, model_key)
    # New experiments persist every matrix cell. Only older CSV files need the
    # reconstruction fallback, which depends on the current processed dataset.
    subject_epoch_counts = (
        {}
        if all(all(row.get(field) not in (None, "") for field in CONFUSION_MATRIX_FIELDS) for row in rows)
        else load_subject_epoch_counts(experiment_dir)
    )
    columns = 3
    row_count = math.ceil(len(rows) / columns)
    output_width_px, output_height_px, output_dpi = 6505, 6593, 300
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(output_width_px / output_dpi, output_height_px / output_dpi),
        squeeze=False,
    )
    protocol_label = PROTOCOL_LABELS_SL.get(protocol_title, protocol_title)
    figure.suptitle(
        f"{protocol_label} - {MODEL_LABELS_SL[model_key]}\n"
        "Matrike zmot na ravni epoh za vse zunanje korake",
        fontsize=30,
        fontweight="bold",
        y=0.982,
    )

    for axis, metrics in zip(axes.flat, rows):
        matrix = infer_confusion_matrix(metrics, subject_epoch_counts)
        image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=18)
        for row_index, column_index in np.ndindex(matrix.shape):
            value = matrix[row_index, column_index]
            color = "white" if value < matrix.max() * 0.45 else "black"
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=30,
                fontweight="bold",
                color=color,
        )
        subject_label = metrics.get("test_subjects") or metrics["test_subject"]
        subject_noun = "udeleženca" if "," in subject_label else "udeleženec"
        axis.set_title(f"Korak {metrics['fold']}, {subject_noun} {subject_label}", fontsize=26, pad=14)
        axis.set_xticks((0, 1), ("Pred", "Po"), fontsize=24)
        axis.set_yticks((0, 1), ("Pred", "Po"), fontsize=24)
        axis.set_xlabel("Napovedani razred", fontsize=25, labelpad=14)
        axis.set_ylabel("Pravi razred", fontsize=25, labelpad=10)

    for axis in axes.flat[len(rows) :]:
        axis.set_visible(False)

    figure.subplots_adjust(
        left=0.065,
        right=0.975,
        bottom=0.085,
        top=0.900,
        hspace=0.62,
        wspace=0.42,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    return output_path


def final_confusion_matrix_paths(experiment_dir: Path) -> list[Path]:
    paths = []
    for model_key in MODEL_ORDER:
        path = experiment_dir / "plots" / model_key / "final_confusion_matrix" / "final_confusion_matrix_epoch_level.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing final confusion matrix: {path}")
        paths.append(path)
    return paths


def write_inventory(rows: list[dict[str, str]], output_dir: Path) -> Path:
    path = output_dir / "figure_inventory.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("section", "protocol", "model", "file", "description"))
        writer.writeheader()
        writer.writerows(rows)
    return path


def split_for_appendix(image_paths: list[Path]) -> list[list[Path]]:
    """Split fold plots into page-sized sheets that remain readable in a thesis."""
    if len(image_paths) <= 4:
        return [image_paths]
    if len(image_paths) == 5:
        return [image_paths[:3], image_paths[3:]]
    return [image_paths[index : index + 4] for index in range(0, len(image_paths), 4)]


def main() -> None:
    args = parse_args()
    experiments = {
        "LOSO": args.loso_experiment,
        "GroupKFold": args.groupkfold_experiment,
    }
    for protocol, path in experiments.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{protocol} experiment directory does not exist: {path}")

    inventory: list[dict[str, str]] = []
    for protocol_key, protocol_title, filename_key in PROTOCOLS:
        experiment_dir = experiments[protocol_key]
        learning_curve_path = save_mean_learning_curves(
            experiment_dir, protocol_title, filename_key, args.output_dir
        )
        inventory.append(
            {
                "section": "Main text",
                "protocol": protocol_key,
                "model": "All models",
                "file": str(learning_curve_path),
                "description": "Mean train/validation accuracy and loss across outer folds (±1 SD).",
            }
        )

        confusion_path = save_contact_sheet(
            final_confusion_matrix_paths(experiment_dir),
            f"{protocol_title}: pooled epoch-level confusion matrices",
            args.output_dir / "main_text" / f"pooled_confusion_matrices_{filename_key}.png",
            columns=4,
            labels=[MODEL_LABELS[model_key] for model_key in MODEL_ORDER],
        )
        inventory.append(
            {
                "section": "Main text",
                "protocol": protocol_key,
                "model": "All models",
                "file": str(confusion_path),
                "description": "Pooled epoch-level confusion matrices, one panel per model.",
            }
        )

        for model_key in MODEL_ORDER:
            plot_root = experiment_dir / "plots" / model_key
            training_paths = sort_by_fold((plot_root / "training_history").glob("training_progress_fold_*.png"))
            training_sheet = save_contact_sheet(
                training_paths,
                f"{protocol_title} – {MODEL_LABELS[model_key]}: training curves for all outer folds",
                args.output_dir / "appendix" / f"{filename_key}_{model_key}_all_fold_learning_curves.png",
            )
            inventory.append(
                {
                    "section": "Appendix",
                    "protocol": protocol_key,
                    "model": MODEL_LABELS[model_key],
                    "file": str(training_sheet),
                    "description": "Train/validation accuracy and loss for every outer fold.",
                }
            )

            confusion_paths = sort_by_fold((plot_root / "confusion_matrices").glob("epoch_fold_*.png"))
            confusion_sheet = save_contact_sheet(
                confusion_paths,
                f"{protocol_title} – {MODEL_LABELS[model_key]}: epoch-level confusion matrices for all outer folds",
                args.output_dir / "appendix" / f"{filename_key}_{model_key}_all_fold_confusion_matrices.png",
                columns=2,
                tile_size=(960, 645),
            )
            inventory.append(
                {
                    "section": "Appendix",
                    "protocol": protocol_key,
                    "model": MODEL_LABELS[model_key],
                    "file": str(confusion_sheet),
                    "description": "Epoch-level confusion matrix for every outer fold.",
                }
            )

            full_size_confusion_sheet = save_large_text_confusion_sheet(
                experiment_dir,
                model_key,
                protocol_title,
                args.output_dir
                / "appendix"
                / "large_confusion_matrices"
                / f"{filename_key}_{model_key}_confusion_matrices_all_folds_3_columns_large_text_sl_6505x6593_title_visible.png",
            )
            inventory.append(
                {
                    "section": "Appendix",
                    "protocol": protocol_key,
                    "model": MODEL_LABELS[model_key],
                    "file": str(full_size_confusion_sheet),
                    "description": "All epoch-level confusion matrices in one high-resolution, three-column sheet with enlarged labels.",
                }
            )

            confusion_chunks = split_for_appendix(confusion_paths)
            for part_index, confusion_chunk in enumerate(confusion_chunks, start=1):
                confusion_sheet = save_contact_sheet(
                    confusion_chunk,
                    f"{protocol_title} - {MODEL_LABELS[model_key]}: epoch-level confusion matrices "
                    f"(appendix part {part_index}/{len(confusion_chunks)})",
                    args.output_dir
                    / "appendix"
                    / "large_confusion_matrices"
                    / f"{filename_key}_{model_key}_confusion_matrices_part_{part_index}.png",
                    columns=min(2, len(confusion_chunk)),
                    tile_size=(1200, 1000),
                )
                inventory.append(
                    {
                        "section": "Appendix",
                        "protocol": protocol_key,
                        "model": MODEL_LABELS[model_key],
                        "file": str(confusion_sheet),
                        "description": "Large, page-sized epoch-level confusion matrices for outer folds "
                        f"in appendix part {part_index}/{len(confusion_chunks)}.",
                    }
                )

    inventory_path = write_inventory(inventory, args.output_dir)
    print(f"Created {len(inventory)} figure summaries in: {args.output_dir}")
    print(f"Figure inventory: {inventory_path}")


if __name__ == "__main__":
    main()
