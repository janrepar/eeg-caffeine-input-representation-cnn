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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


MODEL_ORDER = ("raw_eegnetlike", "features1d_cnn", "features2d_cnn", "hybrid_cnn")
MODEL_LABELS = {
    "raw_eegnetlike": "Raw EEGNet-like",
    "features1d_cnn": "1D feature CNN",
    "features2d_cnn": "2D feature CNN",
    "hybrid_cnn": "Hybrid CNN",
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
) -> Path:
    """Make a legible contact sheet while retaining the original plot labels."""
    if not image_paths:
        raise FileNotFoundError(f"No images available for contact sheet: {title}")
    if labels is not None and len(labels) != len(image_paths):
        raise ValueError("When supplied, labels must match the number of images.")

    tile_size = (640, 430)
    header_height, label_height, margin = 95, 34, 20
    rows = math.ceil(len(image_paths) / columns)
    width = 2 * margin + columns * tile_size[0]
    height = header_height + margin + rows * (tile_size[1] + label_height + margin)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = get_font(32)
    label_font = get_font(20)
    draw.text((margin, 25), title, fill="black", font=title_font)

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

    inventory_path = write_inventory(inventory, args.output_dir)
    print(f"Created {len(inventory)} figure summaries in: {args.output_dir}")
    print(f"Figure inventory: {inventory_path}")


if __name__ == "__main__":
    main()
