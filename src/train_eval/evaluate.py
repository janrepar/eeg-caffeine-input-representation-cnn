from collections import defaultdict, Counter

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)


@torch.no_grad()
def predict_binary_model(model, X, device):
    """
    Predicts probabilities and class labels for binary classification.

    Model returns two-class logits.
    This function applies softmax and returns the probability of After.

    Returns
    -------
    probs: Probability of class 1, i.e. After.
    preds: Predicted class labels, 0 or 1.
    """

    model.eval()

    X = X.to(device)
    logits = model(X)

    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    probs = probabilities[:, 1]
    preds = np.argmax(probabilities, axis=1)

    return probs, preds


@torch.no_grad()
def predict_multiclass_model(model, X, device):
    """
    Predicts probabilities and labels for a model returning logits.

    Model output: logits shape: (batch_size, 2)

    Returns: y_prob: probability of class 1 / After y_pred: predicted class 0 or 1
    """

    model.eval()

    X = X.to(device)
    logits = model(X)

    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    y_prob = probabilities[:, 1]
    y_pred = np.argmax(probabilities, axis=1)

    return y_prob, y_pred


@torch.no_grad()
def predict_hybrid_multiclass_model(model, X_raw, X_feat, device):
    """
    Predicts probabilities and labels for hybrid model.

    Model output: logits: (batch_size, 2)
    Returns:
        y_prob: probability of class 1 / After
        y_pred: predicted class 0 or 1
    """

    model.eval()

    X_raw = X_raw.to(device)
    X_feat = X_feat.to(device)

    logits = model(X_raw, X_feat)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    y_prob = probabilities[:, 1]
    y_pred = np.argmax(probabilities, axis=1)

    return y_prob, y_pred


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
    Computes binary classification metrics.

    Labels:
        0 = Before
        1 = After
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1])
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan

    return metrics


def aggregate_majority_vote(epoch_predictions, subjects, conditions):
    """
    Aggregates epoch-level predictions to subject-condition level.

    Example:
        subject 419 + Before: predictions over all Before epochs -> majority label

        subject 419 + After: predictions over all After epochs -> majority label
    """

    groups = defaultdict(list)

    for pred, subj, cond in zip(epoch_predictions, subjects, conditions):
        key = (str(subj), str(cond))
        groups[key].append(int(pred))

    results = []

    for (subj, cond), preds in groups.items():
        counter = Counter(preds)
        final_pred = counter.most_common(1)[0][0]

        true_label = 1 if str(cond).strip().lower() == "after" else 0

        results.append({
            "subject": subj,
            "condition": cond,
            "true_label": true_label,
            "predicted_label": final_pred,
            "n_epochs": len(preds),
            "votes_before": counter.get(0, 0),
            "votes_after": counter.get(1, 0)
        })

    return results


def aggregate_mean_probability(epoch_probabilities, subjects, conditions):
    """
    Aggregates epoch-level probabilities to subject-condition level.

    For each subject-condition pair:
        mean probability of After is computed.

    If mean probability >= 0.5:
        predicted label = After
    else:
        predicted label = Before
    """

    groups = defaultdict(list)

    for prob, subj, cond in zip(epoch_probabilities, subjects, conditions):
        key = (str(subj), str(cond))
        groups[key].append(float(prob))

    results = []

    for (subj, cond), probs in groups.items():
        mean_prob = float(np.mean(probs))
        final_pred = int(mean_prob >= 0.5)

        true_label = 1 if str(cond).strip().lower() == "after" else 0

        results.append({
            "subject": subj,
            "condition": cond,
            "true_label": true_label,
            "predicted_label": final_pred,
            "mean_probability_after": mean_prob,
            "n_epochs": len(probs)
        })

    return results


def metrics_from_aggregated_results(results):
    """
    Computes metrics from subject-condition-level aggregation results.
    """

    y_true = np.array([row["true_label"] for row in results])
    y_pred = np.array([row["predicted_label"] for row in results])

    return compute_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None
    )


def print_metric_summary(metrics):
    """
    Prints metrics in a readable way.
    """

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")

    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
