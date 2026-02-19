from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def choose_threshold_by_precision(y_true: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, p)
    best_t = 0.5
    best_recall = -1.0
    for i, t in enumerate(thresholds):
        if precision[i] >= min_precision and recall[i] > best_recall:
            best_recall = recall[i]
            best_t = float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def curves(y_true: np.ndarray, p: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    return (fpr, tpr), (rec, prec)
