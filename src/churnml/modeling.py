from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_logreg() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=300)),
        ]
    )


def build_catboost(seed: int = 42) -> Pipeline:
    clf = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
    )
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("clf", clf),
        ]
    )


def calibrate_model(model: Pipeline, method: str, X, y) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(model, method=method, cv=3).fit(X, y)
