import sys
import yaml
from pathlib import Path
from datetime import datetime

# Make project root available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.helpers import load_config, load_used_config, get_nested


def find_latest_result_dir(results_root: Path, model_prefix: str) -> Path | None:
    """
    Finds latest result directory for a given model prefix.

    Example directories:
        raw_eegnetlike_2026-05-20_193208
        features1d_cnn_2026-05-20_193208
        features2d_cnn_2026-05-20_193208
        hybrid_cnn_2026-05-20_193208
    """

    candidates = [
        path for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith(model_prefix)
    ]

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


def summarize_metric(series: pd.Series):
    values = series.dropna().astype(float)

    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    return (
        values.mean(),
        values.std(),
        values.min(),
        values.max(),
    )


def load_model_results(results_root: Path, model_name: str, model_prefix: str, csv_filename: str):
    latest_dir = find_latest_result_dir(results_root, model_prefix)

    if latest_dir is None:
        print(f"No result directory found for model prefix: {model_prefix}")
        return None, None, {}

    used_config = load_used_config(latest_dir)

    csv_path = latest_dir / csv_filename

    if not csv_path.exists():
        print(f"CSV not found for {model_name}: {csv_path}")
        return latest_dir, None, used_config

    df = pd.read_csv(csv_path)
    df["model_name"] = model_name
    df["result_dir"] = str(latest_dir)

    return latest_dir, df, used_config


def make_analysis_dir_name(validation_method, n_validation_subjects, timestamp):
    method_slug = str(validation_method).lower()

    if n_validation_subjects is None:
        val_subjects_part = "unknownvalsubj"
    else:
        val_subjects_part = f"nvalsubj{n_validation_subjects}"

    return f"model_comparison_{method_slug}_{val_subjects_part}_{timestamp}"


def write_used_configs_txt(model_configs: list, output_path: Path):
    """
    Writes all config_used.yaml contents into one txt file.
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON - USED CONFIGS\n")
        f.write("=" * 100)
        f.write("\n\n")

        for item in model_configs:
            model_name = item["model_name"]
            result_dir = item["result_dir"]
            used_config = item["used_config"]

            f.write("=" * 100)
            f.write("\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"RESULT DIR: {result_dir}\n")
            f.write("=" * 100)
            f.write("\n\n")

            if used_config:
                f.write(yaml.safe_dump(
                    used_config,
                    sort_keys=False,
                    allow_unicode=True
                ))
            else:
                f.write("config_used.yaml not found or empty.\n")

            f.write("\n\n")


def main():
    config = load_config("config.yaml")
    results_root = Path("outputs/results")

    validation_method = config["validation"].get("method", "LOSO").lower()
    validation_method_title = validation_method.upper()

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    models = [
        {
            "model_name": "Raw EEGNet-like CNN",
            "model_prefix": "raw_eegnetlike",
            "csv_filename": f"raw_eegnetlike_{validation_method}_metrics.csv",
        },
        {
            "model_name": "Features1D CNN",
            "model_prefix": "features1d_cnn",
            "csv_filename": f"features1d_cnn_{validation_method}_metrics.csv",
        },
        {
            "model_name": "Features2D CNN",
            "model_prefix": "features2d_cnn",
            "csv_filename": f"features2d_cnn_{validation_method}_metrics.csv",
        },
        {
            "model_name": "Hybrid Raw + Features CNN",
            "model_prefix": "hybrid_cnn",
            "csv_filename": f"hybrid_cnn_{validation_method}_metrics.csv",
        },
    ]

    all_fold_dfs = []
    summary_rows = []
    model_configs = []
    first_used_config = None

    for model in models:
        model_name = model["model_name"]
        model_prefix = model["model_prefix"]
        csv_filename = model["csv_filename"]

        latest_dir, df, used_config = load_model_results(
            results_root=results_root,
            model_name=model_name,
            model_prefix=model_prefix,
            csv_filename=csv_filename,
        )

        if used_config and first_used_config is None:
            first_used_config = used_config

        if latest_dir is not None:
            model_configs.append({
                "model_name": model_name,
                "result_dir": str(latest_dir),
                "used_config": used_config,
            })

        if df is None:
            continue

        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)
        print("Result dir:", latest_dir)
        print(df.head())

        all_fold_dfs.append(df)

        metric_columns = [
            "epoch_accuracy",
            "epoch_precision",
            "epoch_recall",
            "epoch_f1",
            "epoch_roc_auc",
        ]

        summary_row = {
            "model_name": model_name,
            "result_dir": str(latest_dir),
            "n_folds": len(df),

            "validation_method": get_nested(used_config, ["validation", "method"], None),
            "n_splits": get_nested(used_config, ["validation", "n_splits"], None),
            "use_validation_subject": get_nested(used_config, ["validation", "use_validation_subject"], None),
            "n_validation_subjects": get_nested(used_config, ["validation", "n_validation_subjects"], None),
            "validation_seed": get_nested(used_config, ["validation", "validation_seed"], None),

            "epochs": get_nested(used_config, ["training", "epochs"], None),
            "batch_size": get_nested(used_config, ["training", "batch_size"], None),
            "learning_rate": get_nested(used_config, ["training", "learning_rate"], None),
            "weight_decay": get_nested(used_config, ["training", "weight_decay"], None),
            "patience": get_nested(used_config, ["training", "patience"], None),
            "validation_frequency": get_nested(used_config, ["training", "validation_frequency"], None),
            "early_stopping_metric": get_nested(used_config, ["training", "early_stopping_metric"], None),
            "label_smoothing": get_nested(used_config, ["training", "label_smoothing"], None),
        }

        for metric in metric_columns:
            if metric in df.columns:
                mean_value, std_value, min_value, max_value = summarize_metric(df[metric])

                summary_row[f"{metric}_mean"] = mean_value
                summary_row[f"{metric}_std"] = std_value
                summary_row[f"{metric}_min"] = min_value
                summary_row[f"{metric}_max"] = max_value
            else:
                summary_row[f"{metric}_mean"] = np.nan
                summary_row[f"{metric}_std"] = np.nan
                summary_row[f"{metric}_min"] = np.nan
                summary_row[f"{metric}_max"] = np.nan

        summary_rows.append(summary_row)

    if not all_fold_dfs:
        raise RuntimeError("No model result CSV files were loaded.")

    all_folds_df = pd.concat(all_fold_dfs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    if first_used_config is not None:
        comparison_validation_method = get_nested(
            first_used_config,
            ["validation", "method"],
            validation_method,
        )
        comparison_n_splits = get_nested(
            first_used_config,
            ["validation", "n_splits"],
            None,
        )
        comparison_n_validation_subjects = get_nested(
            first_used_config,
            ["validation", "n_validation_subjects"],
            None,
        )
    else:
        comparison_validation_method = validation_method
        comparison_n_splits = config["validation"].get("n_splits", None)
        comparison_n_validation_subjects = config["validation"].get("n_validation_subjects", None)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    analysis_dir_name = make_analysis_dir_name(
        validation_method=comparison_validation_method,
        n_validation_subjects=comparison_n_validation_subjects,
        timestamp=timestamp,
    )

    analysis_dir = Path("outputs/analysis") / analysis_dir_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_folds_output_path = analysis_dir / "all_model_fold_results.csv"
    summary_output_path = analysis_dir / "model_comparison_summary.csv"
    used_configs_output_path = analysis_dir / "model_comparison_used_configs.txt"

    all_folds_df.to_csv(all_folds_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    write_used_configs_txt(
        model_configs=model_configs,
        output_path=used_configs_output_path,
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    display_columns = [
        "model_name",
        "validation_method",
        "n_folds",
        "n_splits",
        "n_validation_subjects",
        "use_validation_subject",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "early_stopping_metric",
        "epoch_accuracy_mean",
        "epoch_accuracy_std",
        "epoch_f1_mean",
        "epoch_f1_std",
        "epoch_roc_auc_mean",
        "epoch_roc_auc_std",
    ]

    existing_display_columns = [
        col for col in display_columns
        if col in summary_df.columns
    ]

    print(summary_df[existing_display_columns].to_string(index=False))

    print("\nSaved:")
    print(all_folds_output_path)
    print(summary_output_path)
    print(used_configs_output_path)

    plot_model_comparison_bar(
        summary_df=summary_df,
        metric_mean_col="epoch_accuracy_mean",
        metric_std_col="epoch_accuracy_std",
        ylabel="Accuracy",
        title=f"{validation_method_title} model comparison - Accuracy",
        output_path=analysis_dir / "model_comparison_accuracy.png",
    )

    plot_model_comparison_bar(
        summary_df=summary_df,
        metric_mean_col="epoch_f1_mean",
        metric_std_col="epoch_f1_std",
        ylabel="F1-score",
        title=f"{validation_method_title} model comparison - F1-score",
        output_path=analysis_dir / "model_comparison_f1.png",
    )

    plot_model_comparison_bar(
        summary_df=summary_df,
        metric_mean_col="epoch_roc_auc_mean",
        metric_std_col="epoch_roc_auc_std",
        ylabel="ROC-AUC",
        title=f"{validation_method_title} model comparison - ROC-AUC",
        output_path=analysis_dir / "model_comparison_roc_auc.png",
    )

    plot_metric_by_subject(
        all_folds_df=all_folds_df,
        metric_col="epoch_accuracy",
        title=f"{validation_method_title} accuracy by test subject",
        output_path=analysis_dir / "accuracy_by_subject.png",
    )

    plot_metric_by_subject(
        all_folds_df=all_folds_df,
        metric_col="epoch_f1",
        title=f"{validation_method_title} F1-score by test subject",
        output_path=analysis_dir / "f1_by_subject.png",
    )


def plot_model_comparison_bar(summary_df, metric_mean_col, metric_std_col, ylabel, title, output_path):
    df = summary_df.copy()

    if metric_mean_col not in df.columns:
        print(f"Skipping plot. Missing column: {metric_mean_col}")
        return

    df = df.sort_values(metric_mean_col, ascending=False)

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))
    means = df[metric_mean_col].values
    stds = df[metric_std_col].values if metric_std_col in df.columns else None

    plt.bar(x, means, yerr=stds, capsize=6)
    plt.axhline(0.5, linestyle="--", color='red', linewidth=2, label="Chance level")

    plt.xticks(x, df["model_name"].values, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_metric_by_subject(all_folds_df, metric_col, title, output_path):
    if metric_col not in all_folds_df.columns:
        print(f"Skipping plot. Missing column: {metric_col}")
        return

    df = all_folds_df.copy()

    if "test_subject" not in df.columns:
        print("Skipping subject plot. Missing column: test_subject")
        return

    pivot_df = df.pivot_table(
        index="test_subject",
        columns="model_name",
        values=metric_col,
        aggfunc="mean",
    )

    pivot_df = pivot_df.sort_index()

    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", figsize=(12, 6))

    plt.axhline(0.5, linestyle="--", color='red', linewidth=2, label="Chance level")

    plt.ylabel(metric_col)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()