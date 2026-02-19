import argparse
import os

from churnml.data import generate_synthetic_churn, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=6000)
    parser.add_argument("--prod_size", type=int, default=3000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = generate_synthetic_churn(
        n=args.train_size,
        seed=args.seed,
        drift=0.0,
    )

    prod_df = generate_synthetic_churn(
        n=args.prod_size,
        seed=args.seed + 1,
        drift=0.35,
    )

    save_csv(train_df, os.path.join(args.out_dir, "train.csv"))
    save_csv(prod_df, os.path.join(args.out_dir, "next_month.csv"))

    print("Saved:")
    print(os.path.join(args.out_dir, "train.csv"))
    print(os.path.join(args.out_dir, "next_month.csv"))


if __name__ == "__main__":
    main()
