from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["tenure", "monthly_charges", "support_calls", "late_payments"]
CAT_COLS = ["contract"]
BIN_COLS = ["has_internet", "has_tv"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("bin", "passthrough", BIN_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["churn"])
    y = df["churn"].astype(int).to_numpy()
    return X, y
