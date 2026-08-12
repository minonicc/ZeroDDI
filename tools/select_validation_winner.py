"""Create an auditable validation-only experiment winner decision."""

import argparse
import csv
import json
from pathlib import Path


PRIMARY = ("ACC", "Kappa", "Macro-F1")


def load_complete_summary(path, expected_epochs):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Validation summary is empty")
    names = [row["experiment"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Validation summary contains duplicate experiments")
    parsed = []
    for row in rows:
        completed = int(row["completed_epochs"])
        if completed != expected_epochs:
            raise ValueError(
                f"{row['experiment']} completed {completed} epochs; "
                f"expected {expected_epochs}"
            )
        parsed.append(
            {
                "experiment": row["experiment"],
                "best_epoch": int(row["epoch"]),
                "completed_epochs": completed,
                "log": row["log"],
                **{metric: float(row[metric]) for metric in PRIMARY},
            }
        )
    return parsed


def decide(rows, macro_f1_tolerance=0.001):
    if not rows:
        raise ValueError("No validation runs were provided")
    rows = sorted(rows, key=lambda row: row["experiment"])
    max_macro = max(row["Macro-F1"] for row in rows)
    contenders = [
        row
        for row in rows
        if max_macro - row["Macro-F1"] < macro_f1_tolerance
    ]
    winner = sorted(
        contenders,
        key=lambda row: (
            -row["Kappa"],
            -row["ACC"],
            -row["Macro-F1"],
            row["experiment"],
        ),
    )[0]
    comparisons = []
    for row in sorted(rows, key=lambda item: item["experiment"]):
        if row["experiment"] == winner["experiment"]:
            continue
        deltas = {metric: winner[metric] - row[metric] for metric in PRIMARY}
        comparisons.append(
            {
                "reference": row["experiment"],
                "deltas": deltas,
                "macro_gain_with_both_acc_kappa_down": (
                    deltas["Macro-F1"] > 0
                    and deltas["Kappa"] < 0
                    and deltas["ACC"] < 0
                ),
            }
        )
    return {
        "metric_split": "validation",
        "selection_rule": (
            "highest Macro-F1 band with max-minus-candidate < 0.001; "
            "within band use Kappa then ACC"
        ),
        "macro_f1_tolerance": macro_f1_tolerance,
        "max_macro_f1": max_macro,
        "contenders": sorted(row["experiment"] for row in contenders),
        "winner": winner,
        "comparisons": comparisons,
        "requires_tradeoff_review": any(
            comparison["macro_gain_with_both_acc_kappa_down"]
            for comparison in comparisons
        ),
        "all_runs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_epochs <= 0:
        raise ValueError("--expected-epochs must be positive")
    rows = load_complete_summary(args.summary, args.expected_epochs)
    decision = decide(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validation winner: {decision['winner']['experiment']} "
        f"(tradeoff_review={decision['requires_tradeoff_review']})"
    )


if __name__ == "__main__":
    main()
