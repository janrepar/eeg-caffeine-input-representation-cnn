import numpy as np

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