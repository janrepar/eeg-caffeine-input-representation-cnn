import numpy as np
import yaml
from pathlib import Path


def get_scalar(value):
    """
    Pomaga pri branju MATLAB/EEGLAB polj iz scipy.io.loadmat.
    """
    while isinstance(value, np.ndarray):
        value = value[0]
    return value


def condition_to_label(condition: str) -> int:
    """
    Condition to binary:
    Before = 0
    After = 1
    """
    condition = str(condition).lower()

    if condition == "before":
        return 0
    if condition == "after":
        return 1

    raise ValueError(f"Unknown condition: {condition}")


def label_to_condition(label: int) -> str:
    if int(label) == 0:
        return "Before"

    if int(label) == 1:
        return "After"

    raise ValueError(f"Unknown label: {label}")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def load_used_config(result_dir: Path) -> dict:
    """
    Loads config_used.yaml from a model result directory.
    Returns empty dict if config is missing.
    """

    config_path = result_dir / "config_used.yaml"

    if not config_path.exists():
        print(f"config_used.yaml not found: {config_path}")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(config: dict, keys: list, default=None):
    """
    Safely reads nested values from dict.
    Example:
        get_nested(config, ["validation", "method"])
    """

    current = config

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current