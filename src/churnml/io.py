from __future__ import annotations

import json
import os
from typing import Any

import joblib


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_model(model: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
