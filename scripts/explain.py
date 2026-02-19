import argparse
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnml.data import load_csv
from churnml.features import split_xy


def _sample(X, n, seed=42):
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=seed)


def _get_pipeline_parts(model):
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    return prep, clf


def _feature_names(prep):
    try:
        return prep.get_feature_names_out().tolist()
    except Exception:
        return [
            f"f{i}"
            for i in range(prep.transform(np.zeros((1, len(prep.feature_names_in_)))).shape[1])
        ]


def _to_dense(a):
    try:
        return a.toarray()
    except Exception:
        return np.asarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--model", choices=["catboost", "logreg"], default="catboost")
    ap.add_argument("--n_background", type=int, default=500)
    ap.add_argument("--n_explain", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(os.path.join(args.data_dir, "train.csv"))
    X, y = split_xy(df)

    if args.model == "catboost":
        model_path = os.path.join(args.artifacts_dir, "catboost_model.joblib")
    else:
        model_path = os.path.join(args.artifacts_dir, "model.joblib")

    model = joblib.load(model_path)
    prep, clf = _get_pipeline_parts(model)

    X_bg = _sample(X, args.n_background)
    X_ex = _sample(X, args.n_explain, seed=43)

    Z_bg = _to_dense(prep.transform(X_bg))
    Z_ex = _to_dense(prep.transform(X_ex))
    names = _feature_names(prep)

    if args.model == "catboost":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Z_ex)
        values = sv[1] if isinstance(sv, list) else sv
    else:
        explainer = shap.LinearExplainer(clf, Z_bg, feature_perturbation="interventional")
        values = explainer.shap_values(Z_ex)

    plt.figure()
    shap.summary_plot(values, Z_ex, feature_names=names, show=False, max_display=18)
    out_path = os.path.join(args.out_dir, f"shap_summary_{args.model}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()

    print(out_path)


if __name__ == "__main__":
    main()
