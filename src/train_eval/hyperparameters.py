"""Resolve and record model hyperparameters for final evaluation runs."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TRAINING_PARAMETER_KEYS = {"learning_rate", "weight_decay", "label_smoothing"}


def _mode(values: list[Any]) -> Any:
    """Return a deterministic mode; use the textual representation to break ties."""
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]


def aggregate_loso_hyperparameters(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate fold-specific LOSO Optuna optima into one GroupKFold configuration.

    Continuous parameters are aggregated with their median. Integer architecture
    parameters are aggregated with their mode. This keeps a GroupKFold experiment
    fixed across its outer folds while preserving a transparent link to LOSO HPO.
    """
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"No LOSO Optuna records found in {summary_path}")

    parameter_sets = [record.get("best_params") for record in records]
    if any(not isinstance(params, dict) for params in parameter_sets):
        raise ValueError(f"Invalid best_params entries in {summary_path}")

    parameter_names = set(parameter_sets[0])
    if any(set(params) != parameter_names for params in parameter_sets[1:]):
        raise ValueError(f"LOSO Optuna folds use inconsistent parameter sets in {summary_path}")

    aggregated: dict[str, Any] = {}
    aggregation_by_parameter: dict[str, str] = {}

    for name in sorted(parameter_names):
        values = [params[name] for params in parameter_sets]
        if all(isinstance(value, bool) for value in values):
            aggregated[name] = _mode(values)
            aggregation_by_parameter[name] = "mode"
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            aggregated[name] = _mode(values)
            aggregation_by_parameter[name] = "mode"
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            aggregated[name] = float(np.median(values))
            aggregation_by_parameter[name] = "median"
        else:
            aggregated[name] = _mode(values)
            aggregation_by_parameter[name] = "mode"

    metadata = {
        "source_summary": str(summary_path),
        "n_loso_folds": len(records),
        "outer_test_subjects": [record.get("outer_test_subject") for record in records],
        "aggregation_by_parameter": aggregation_by_parameter,
        "aggregation_rule": "median for continuous parameters; mode for integer/categorical parameters",
    }
    return aggregated, metadata


def load_groupkfold_loso_aggregate(config: dict[str, Any], model_key: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load and materialise the global GroupKFold configuration derived from LOSO HPO."""
    analysis_type = config.get("experiment", {}).get("analysis_type")
    if analysis_type != "caffeine_before_vs_after":
        raise ValueError(
            "LOSO Optuna summaries currently contain Caffeine-only optimisation results. "
            "They can only be transferred to GroupKFold for caffeine_before_vs_after."
        )

    output_dir = Path(config["outputs"]["output_dir"])
    hpo_dir = output_dir / "hyperparameter_optimization" / model_key
    summary_path = hpo_dir / "nested_optimization_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing LOSO Optuna summary for GroupKFold hyperparameter transfer: {summary_path}"
        )

    parameters, metadata = aggregate_loso_hyperparameters(summary_path)
    artifact_path = hpo_dir / "global_params_from_loso.json"
    artifact_path.write_text(
        json.dumps(
            {
                "parameter_source": "LOSO Optuna aggregate",
                "best_params": parameters,
                **metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return parameters, metadata, artifact_path


def apply_hyperparameters(
    base_model_config: dict[str, Any],
    base_training_config: dict[str, Any],
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return independent model and training configurations with HPO overrides."""
    model_config = copy.deepcopy(base_model_config)
    training_config = copy.deepcopy(base_training_config)

    for name, value in parameters.items():
        if name in TRAINING_PARAMETER_KEYS:
            training_config[name] = value
        else:
            model_config[name] = value

    return model_config, training_config


def write_hyperparameter_manifest(
    results_dir: Path,
    model_name: str,
    validation_method: str,
    folds: list[dict[str, Any]],
    groupkfold_aggregate: dict[str, Any] | None = None,
) -> Path:
    """Write exact per-fold effective configurations for reproducibility."""
    payload: dict[str, Any] = {
        "model_name": model_name,
        "validation_method": validation_method,
        "folds": folds,
    }
    if groupkfold_aggregate is not None:
        payload["groupkfold_loso_aggregate"] = groupkfold_aggregate

    manifest_path = results_dir / "fold_hyperparameters.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
