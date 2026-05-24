import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_multiclass_model(model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
   device: torch.device,
   epochs: int = 100,
   batch_size: int = 32,
   learning_rate: float = 1e-3,
   weight_decay: float = 1e-4,
   patience: int = 15,
   validation_frequency: int = 30,
   early_stopping_metric='val_loss',
   label_smoothing: float = 0.0
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Trains a 2-class / multiclass classification model.

    The model should return raw logits of shape: (batch_size, n_classes)

    For binary Before/After classification: n_classes = 2

    This function uses CrossEntropyLoss.
    Do not apply Softmax inside the model during training.
    """

    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print(f"Early stopping metric: {early_stopping_metric}")
    print(f"Label smoothing: {label_smoothing}")

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

    if early_stopping_metric == "val_loss":
        best_score = float("inf")
        mode = "min"
    elif early_stopping_metric == "val_acc":
        best_score = -float("inf")
        mode = "max"
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")

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
            batch_y = batch_y.to(device).long()

            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

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

        current_score = val_loss if early_stopping_metric == "val_loss" else val_acc

        is_improvement = (
            current_score < best_score
            if mode == "min"
            else current_score > best_score
        )

        if is_improvement:
            if early_stopping_metric == "val_loss":
                best_score = val_loss
            else:
                best_score = val_acc

            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            history["best_epoch"] = epoch + 1
            history["best_train_acc"] = train_acc
            history["best_val_acc"] = val_acc
            history["best_val_loss"] = val_loss
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    model.load_state_dict(best_model_state)

    return model, history


def train_hybrid_multiclass_model(
    model: nn.Module,
    X_raw_train: torch.Tensor,
    X_feat_train: torch.Tensor,
    y_train: torch.Tensor,
    X_raw_val: torch.Tensor,
    X_feat_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    validation_frequency: int = 30,
    early_stopping_metric: str = "val_loss",
    label_smoothing=0.0,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Trains a hybrid model with two inputs:
        raw EEG input
        feature input

    Model should return logits with shape:
        (batch_size, n_classes)

    Uses CrossEntropyLoss.
    """

    model = model.to(device)

    train_dataset = TensorDataset(X_raw_train, X_feat_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print(f"Early stopping metric: {early_stopping_metric}")
    print(f"Label smoothing: {label_smoothing}")

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
        "best_epoch": None,
    }

    if early_stopping_metric == "val_loss":
        best_score = float("inf")
        mode = "min"
    elif early_stopping_metric == "val_acc":
        best_score = -float("inf")
        mode = "max"
    else:
        raise ValueError(f"Unknown early_stopping_metric: {early_stopping_metric}")

    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    iteration = 0

    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0.0
        correct_train = 0
        total_train = 0

        for batch_raw, batch_feat, batch_y in train_loader:
            iteration += 1

            batch_raw = batch_raw.to(device)
            batch_feat = batch_feat.to(device)
            batch_y = batch_y.to(device).long()

            optimizer.zero_grad()

            logits = model(batch_raw, batch_feat)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

            batch_acc = (preds.cpu() == batch_y.cpu()).float().mean().item()

            train_loss_sum += loss.item() * batch_raw.size(0)
            correct_train += (preds.cpu() == batch_y.cpu()).sum().item()
            total_train += batch_raw.size(0)

            history["iter"].append(iteration)
            history["iter_train_loss"].append(loss.item())
            history["iter_train_acc"].append(batch_acc)

            if iteration % validation_frequency == 0:
                val_loss_iter, val_acc_iter = evaluate_hybrid_loss_and_accuracy(
                    model=model,
                    X_raw=X_raw_val,
                    X_feat=X_feat_val,
                    y=y_val,
                    criterion=criterion,
                    device=device,
                )

                history["iter_val_iter"].append(iteration)
                history["iter_val_loss"].append(val_loss_iter)
                history["iter_val_acc"].append(val_acc_iter)

        train_loss = train_loss_sum / total_train
        train_acc = correct_train / total_train

        val_loss, val_acc = evaluate_hybrid_loss_and_accuracy(
            model=model,
            X_raw=X_raw_val,
            X_feat=X_feat_val,
            y=y_val,
            criterion=criterion,
            device=device,
        )

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

        current_score = val_loss if early_stopping_metric == "val_loss" else val_acc

        is_improvement = (
            current_score < best_score
            if mode == "min"
            else current_score > best_score
        )

        if is_improvement:
            if early_stopping_metric == "val_loss":
                best_score = val_loss
            else:
                best_score = val_acc

            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            history["best_epoch"] = epoch + 1
            history["best_train_acc"] = train_acc
            history["best_val_acc"] = val_acc
            history["best_val_loss"] = val_loss
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
    y = y.to(device).long()

    logits = model(X)
    loss = criterion(logits, y)

    preds = torch.argmax(logits, dim=1)
    acc = (preds.cpu() == y.cpu()).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def evaluate_hybrid_loss_and_accuracy(
    model: nn.Module,
    X_raw: torch.Tensor,
    X_feat: torch.Tensor,
    y: torch.Tensor,
    criterion,
    device: torch.device,
):
    model.eval()

    X_raw = X_raw.to(device)
    X_feat = X_feat.to(device)
    y = y.to(device).long()

    logits = model(X_raw, X_feat)
    loss = criterion(logits, y)

    preds = torch.argmax(logits, dim=1)
    acc = (preds.cpu() == y.cpu()).float().mean().item()

    return loss.item(), acc