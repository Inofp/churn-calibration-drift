import argparse
import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnml.data import load_csv
from churnml.features import split_xy
from churnml.modeling import calibrate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--in_dir", default="artifacts")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--method", choices=["sigmoid", "isotonic"], default="isotonic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(os.path.join(args.data_dir, "train.csv"))
    X, y = split_xy(df)

    base = joblib.load(os.path.join(args.in_dir, "model.joblib"))
    cal = calibrate_model(base, args.method, X, y)

    out_path = os.path.join(args.out_dir, f"calibrated_{args.method}.joblib")
    joblib.dump(cal, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
