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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "LOG"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = []
    for name, path in args.run:
        epochs = parse_log(path)
        best = None
        for row in epochs:
            if better(row, best):
                best = row
        if best is None:
            raise RuntimeError(f"No completed validation epoch in {path}")
        summary.append({"experiment": name, "log": path, **best})

    with open(args.output, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
