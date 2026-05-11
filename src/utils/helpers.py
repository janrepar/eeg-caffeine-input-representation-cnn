import numpy as np
import yaml


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


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads YAML configuration file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config