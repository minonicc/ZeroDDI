"""Summarize validation metrics without consulting test results."""

import argparse
import csv
import re


EPOCH_RE = re.compile(r"epoch is (\d+) \|\| Train batch_loss is ([0-9.eE+-]+)")
METRIC_RE = re.compile(
    r"ACC:([0-9.eE+-]+), Kappa:([0-9.eE+-]+), Macro-F1:([0-9.eE+-]+)"
)


def parse_log(path):
    rows = []
    current_epoch = None
    train_loss = None
    with open(path, encoding="utf-8") as log_file:
        for line in log_file:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1)) + 1
                train_loss = float(epoch_match.group(2))
                continue
            metric_match = METRIC_RE.search(line)
            if metric_match and current_epoch is not None:
                rows.append(
                    {
                        "epoch": current_epoch,
                        "train_loss": train_loss,
                        "ACC": float(metric_match.group(1)),
                        "Kappa": float(metric_match.group(2)),
                        "Macro-F1": float(metric_match.group(3)),
                    }
                )
                current_epoch = None
    return rows


def better(candidate, incumbent):
    if incumbent is None:
        return True
    delta = candidate["Macro-F1"] - incumbent["Macro-F1"]
    if abs(delta) >= 0.001:
        return delta > 0
    if candidate["Kappa"] != incumbent["Kappa"]:
        return candidate["Kappa"] > incumbent["Kappa"]
    return candidate["ACC"] > incumbent["ACC"]


def format_epoch_ranges(epochs):
    if not epochs:
        return "none"
    ranges = []
    start = previous = epochs[0]
    for epoch in epochs[1:]:
        if epoch != previous + 1:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = epoch
        previous = epoch
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def require_complete_epochs(name, path, epochs, expected_epochs):
    observed = [row["epoch"] for row in epochs]
    expected = list(range(1, expected_epochs + 1))
    if observed != expected:
        observed_set = set(observed)
        missing = [epoch for epoch in expected if epoch not in observed_set]
        unexpected = [epoch for epoch in observed if epoch not in set(expected)]
        raise RuntimeError(
            f"{name} has {len(observed)} validation rows in {path}; expected "
            f"ordered epochs 1..{expected_epochs}. Observed={observed}; "
            f"missing={format_epoch_ranges(missing)}; "
            f"unexpected={format_epoch_ranges(unexpected)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "LOG"), required=True)
    parser.add_argument(
        "--expected-epochs",
        type=int,
        help="fail unless every run contains this many completed validation epochs",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = []
    for name, path in args.run:
        epochs = parse_log(path)
        if args.expected_epochs is not None:
            require_complete_epochs(name, path, epochs, args.expected_epochs)
        best = None
        for row in epochs:
            if better(row, best):
                best = row
        if best is None:
            raise RuntimeError(f"No completed validation epoch in {path}")
        summary.append(
            {
                "experiment": name,
                "log": path,
                "completed_epochs": len(epochs),
                **best,
            }
        )

    with open(args.output, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=summary[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
