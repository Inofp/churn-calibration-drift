import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnml.data import load_csv
from churnml.drift import psi_report
from churnml.features import NUM_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train = load_csv(os.path.join(args.data_dir, "train.csv"))
    prod = load_csv(os.path.join(args.data_dir, "next_month.csv"))

    rep = psi_report(train, prod, numeric_cols=NUM_COLS)
    out_path = os.path.join(args.out_dir, "psi_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
