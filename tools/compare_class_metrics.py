"""Compare final per-DDIE metrics without feeding test results back into tuning."""

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("precision", "recall", "f1")


def load_metrics(path, suffix):
    frame = pd.read_csv(path)
    key = "event_id" if "event_id" in frame.columns else "class_id"
    required = {key, "support", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[key].duplicated().any():
        raise ValueError(f"{path} contains duplicate {key} values")
    columns = [key, "support", *METRICS]
    return key, frame[columns].rename(
        columns={column: f"{column}_{suffix}" for column in columns if column != key}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")

    reference_key, reference = load_metrics(args.reference, "reference")
    candidate_key, candidate = load_metrics(args.candidate, "candidate")
    if reference_key != candidate_key:
        raise ValueError(
            f"metric files use different keys: {reference_key} and {candidate_key}"
        )
    key = reference_key
    comparison = reference.merge(
        candidate, on=key, how="outer", validate="one_to_one", indicator=True
    )
    if not comparison["_merge"].eq("both").all():
        missing = comparison.loc[comparison["_merge"] != "both", [key, "_merge"]]
        raise ValueError(f"class sets differ:\n{missing.to_string(index=False)}")
    comparison = comparison.drop(columns="_merge")
    if (comparison["support_reference"] != comparison["support_candidate"]).any():
        raise ValueError("reference and candidate supports differ; compare the same split")

    support = comparison["support_reference"]
    lower, upper = support.quantile([0.25, 0.75])
    comparison["support_group"] = "middle"
    comparison.loc[support <= lower, "support_group"] = "minority_q1"
    comparison.loc[support >= upper, "support_group"] = "majority_q4"
    for metric in METRICS:
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_candidate"]
            - comparison[f"{metric}_reference"]
        )
    comparison = comparison.sort_values("delta_f1", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, lineterminator="\n")

    display = [key, "support_reference", "support_group", "delta_f1"]
    print("Largest F1 gains:")
    print(comparison.head(args.top)[display].to_string(index=False))
    print("Largest F1 regressions:")
    print(comparison.tail(args.top).sort_values("delta_f1")[display].to_string(index=False))
    print("Mean deltas by support group:")
    print(
        comparison.groupby("support_group")[[f"delta_{metric}" for metric in METRICS]]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
