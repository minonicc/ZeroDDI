import argparse
import os

import pandas as pd


def split_group(group, rng_seed, train_ratio, val_ratio):
    group = group.sample(frac=1.0, random_state=rng_seed)
    n = len(group)

    if n == 1:
        return group, group.iloc[0:0], group.iloc[0:0]
    if n == 2:
        return group.iloc[:1], group.iloc[0:0], group.iloc[1:]
    if n < 10:
        return group.iloc[: n - 2], group.iloc[n - 2 : n - 1], group.iloc[n - 1 :]

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    train = group.iloc[:n_train]
    val = group.iloc[n_train : n_train + n_val]
    test = group.iloc[n_train + n_val :]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Create a standard seen-label DDI event split.")
    parser.add_argument("--input", default="data/DrugBank5.1.9/DDI_final2.csv")
    parser.add_argument("--output-dir", default="data/DrugBank5.1.9/seen_random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train_parts = []
    val_parts = []
    test_parts = []

    for event_id, group in df.groupby("event_id", sort=True):
        train, val, test = split_group(
            group,
            rng_seed=args.seed + int(event_id),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    train_labels = set(train_df["event_id"])
    val_labels = set(val_df["event_id"])
    test_labels = set(test_df["event_id"])

    print(f"input rows: {len(df)}, events: {df['event_id'].nunique()}")
    print(f"train rows: {len(train_df)}, events: {train_df['event_id'].nunique()}")
    print(f"val rows: {len(val_df)}, events: {val_df['event_id'].nunique()}")
    print(f"test rows: {len(test_df)}, events: {test_df['event_id'].nunique()}")
    print(f"val labels not in train: {sorted(val_labels - train_labels)}")
    print(f"test labels not in train: {sorted(test_labels - train_labels)}")


if __name__ == "__main__":
    main()
