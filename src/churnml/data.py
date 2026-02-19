from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_churn(n: int, seed: int, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 60, size=n)

    monthly_charges = rng.normal(35.0 + drift * 3.0, 12.0, size=n).clip(5, 120)
    support_calls = rng.poisson(0.8 + drift * 0.2, size=n).clip(0, 10)
    late_payments = rng.poisson(0.3 + drift * 0.15, size=n).clip(0, 6)

    contract = rng.choice(
        ["month-to-month", "1-year", "2-year"],
        size=n,
        p=[0.55 + drift * 0.05, 0.3 - drift * 0.03, 0.15 - drift * 0.02],
    )

    has_internet = rng.choice([0, 1], size=n, p=[0.25 - drift * 0.02, 0.75 + drift * 0.02])
    has_tv = rng.choice([0, 1], size=n, p=[0.45 - drift * 0.02, 0.55 + drift * 0.02])

    contract_w = np.where(
        contract == "month-to-month", 0.9, np.where(contract == "1-year", 0.2, -0.2)
    )

    z = (
        1.2 * contract_w
        + 0.02 * (monthly_charges - 35.0)
        + 0.35 * support_calls
        + 0.55 * late_payments
        - 0.02 * tenure
        + 0.25 * (1 - has_internet)
        + 0.10 * (1 - has_tv)
        + rng.normal(0.0, 0.6, size=n)
    )

    p = _sigmoid(z)
    churn = (rng.random(n) < p).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "late_payments": late_payments,
            "contract": contract,
            "has_internet": has_internet,
            "has_tv": has_tv,
            "churn": churn,
        }
    )


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
