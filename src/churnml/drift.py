from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0] -= eps
    edges[-1] += eps

    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)

    e = e_counts / max(e_counts.sum(), 1)
    a = a_counts / max(a_counts.sum(), 1)

    e = np.clip(e, eps, 1.0)
    a = np.clip(a, eps, 1.0)

    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(train_df: pd.DataFrame, prod_df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    out = {c: psi(train_df[c].to_numpy(), prod_df[c].to_numpy(), bins=10) for c in numeric_cols}
    out["psi_mean"] = float(np.mean(list(out.values()))) if out else 0.0
    return out
