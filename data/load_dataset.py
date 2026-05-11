import numpy as np


def load_dataset_npz(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]
    conditions = data["conditions"]

    return X, y, subjects, conditions