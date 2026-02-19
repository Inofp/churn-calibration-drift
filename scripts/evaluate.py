import argparse
import json
import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnml.data import load_csv
from churnml.features import split_xy
from churnml.metrics import choose_threshold_by_precision, compute_metrics, curves
from churnml.plots import save_pr_curve, save_reliability, save_roc_curve


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def _eval_variant(
    name: str, model, X, y, out_dir: str, min_precision: float, write_plots: bool
) -> dict:
    p = _proba(model, X)
    t = choose_threshold_by_precision(y, p, min_precision)
    m = compute_metrics(y, p, t)

    if write_plots:
        (fpr, tpr), (rec, prec) = curves(y, p)
        safe = name.replace(":", "_").replace("/", "_")
        save_roc_curve(fpr, tpr, os.path.join(out_dir, f"roc_{safe}.png"))
        save_pr_curve(rec, prec, os.path.join(out_dir, f"pr_{safe}.png"))
        save_reliability(y, p, os.path.join(out_dir, f"reliability_{safe}.png"))

    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--min_precision", type=float, default=0.65)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(os.path.join(args.data_dir, "train.csv"))
    X, y = split_xy(df)

    metrics_out = {}

    base_path = os.path.join(args.artifacts_dir, "model.joblib")
    if os.path.exists(base_path):
        base = joblib.load(base_path)
        metrics_out["logreg/base"] = _eval_variant(
            "logreg_base", base, X, y, args.out_dir, args.min_precision, write_plots=False
        )

    for method in ["sigmoid", "isotonic"]:
        path = os.path.join(args.artifacts_dir, f"calibrated_{method}.joblib")
        if not os.path.exists(path):
            continue
        cal = joblib.load(path)
        metrics_out[f"logreg/calibrated_{method}"] = _eval_variant(
            f"logreg_calibrated_{method}",
            cal,
            X,
            y,
            args.out_dir,
            args.min_precision,
            write_plots=True,
        )

    cb_path = os.path.join(args.artifacts_dir, "catboost_model.joblib")
    if os.path.exists(cb_path):
        cb = joblib.load(cb_path)
        metrics_out["catboost/base"] = _eval_variant(
            "catboost_base", cb, X, y, args.out_dir, args.min_precision, write_plots=True
        )

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    thresholds_path = os.path.join(args.out_dir, "thresholds.json")
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v["threshold"] for k, v in metrics_out.items()}, f, ensure_ascii=False, indent=2
        )

    best_variant = (
        max(metrics_out.items(), key=lambda kv: kv[1]["pr_auc"])[0] if metrics_out else None
    )
    summary = {
        "best_variant": best_variant,
        "variants": list(metrics_out.keys()),
        "min_precision_constraint": float(args.min_precision),
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(metrics_path)
    print(summary_path)


if __name__ == "__main__":
    main()
