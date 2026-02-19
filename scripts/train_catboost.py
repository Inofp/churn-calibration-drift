import argparse
import json
import os
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnml.config import TrainConfig
from churnml.data import load_csv
from churnml.features import split_xy
from churnml.modeling import build_catboost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = TrainConfig()

    df = load_csv(os.path.join(args.data_dir, "train.csv"))
    X, y = split_xy(df)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y,
    )

    model = build_catboost(seed=cfg.seed).fit(X_tr, y_tr)

    out_path = os.path.join(args.out_dir, "catboost_model.joblib")
    joblib.dump(model, out_path)

    with open(os.path.join(args.out_dir, "catboost_meta.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
