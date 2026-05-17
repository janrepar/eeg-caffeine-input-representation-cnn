import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_binary_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device, epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 1e-4, patience: int = 15, validation_frequency: int = 30) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Trains a binary classification model.

    The model should return raw logits, not probabilities.
    Therefore this function uses BCEWithLogitsLoss.

    Parameters
    ----------
    model: PyTorch model.
    X_train: Training input tensor.
    y_train: Training labels. Shape: (n_train). Values: 0 or 1.
    X_val: Validation input tensor.
    y_val: Validation labels. Shape: (n_val). Values: 0 or 1.
    device: CPU or CUDA device.

    Returns
    -------
    model: Model with the best validation loss weights.
    history: Dictionary containing train/validation loss and accuracy.
    """

    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],

        "iter": [],
        "iter_train_loss": [],
        "iter_train_acc": [],

        "iter_val_iter": [],
        "iter_val_loss": [],
        "iter_val_acc": [],

        "best_train_acc": None,
        "best_val_acc": None,
        "best_val_loss": None,
        "best_epoch": None
    }

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    iteration = 0

    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            iteration += 1

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).float()

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            batch_acc = (preds.cpu() == batch_y.cpu().long()).float().mean().item()

            train_loss_sum += loss.item() * batch_X.size(0)
            correct_train += (preds.cpu() == batch_y.cpu().long()).sum().item()
            total_train += batch_X.size(0)

            # save iteration-level training metrics
            history["iter"].append(iteration)
            history["iter_train_loss"].append(loss.item())
            history["iter_train_acc"].append(batch_acc)

            # validation every N iterations
            if iteration % validation_frequency == 0:
                val_loss_iter, val_acc_iter = evaluate_loss_and_accuracy(
                    model=model,
                    X=X_val,
                    y=y_val,
                    criterion=criterion,
                    device=device
                )

                history["iter_val_iter"].append(iteration)
                history["iter_val_loss"].append(val_loss_iter)
                history["iter_val_acc"].append(val_acc_iter)

        train_loss = train_loss_sum / total_train
        train_acc = correct_train / total_train

        val_loss, val_acc = evaluate_loss_and_accuracy(
            model=model,
            X=X_val,
            y=y_val,
            criterion=criterion,
            device=device
        )

        # save epoch-level metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            history["best_train_acc"] = train_acc
            history["best_val_acc"] = val_acc
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch + 1
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)

    return model, history


@torch.no_grad()
def evaluate_loss_and_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion, device: torch.device):
    """
    Evaluates loss and accuracy on a full tensor dataset.
    """

    model.eval()

    X = X.to(device)
    y = y.to(device).float()

    logits = model(X)
    loss = criterion(logits, y)

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    acc = (preds.cpu() == y.cpu().long()).float().mean().item()

    return loss.item(), acc