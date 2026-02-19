from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def save_roc_curve(fpr, tpr, out_path: str) -> None:
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_pr_curve(recall, precision, out_path: str) -> None:
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_reliability(y_true: np.ndarray, p: np.ndarray, out_path: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability diagram")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
