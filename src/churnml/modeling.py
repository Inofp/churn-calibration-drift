from __future__ import annotations

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


def calibrate_model(model: Pipeline, method: str, X, y) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(model, method=method, cv=3).fit(X, y)
