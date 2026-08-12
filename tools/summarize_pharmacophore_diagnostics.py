"""Flatten epoch-level pharmacophore diagnostics into a comparison table."""

import argparse
import csv
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(r"epoch is (\d+) \|\|")
ALPHA_RE = re.compile(r"fixed_substructure_alpha:([0-9.eE+-]+)")
ARRAY_FIELDS = (
    "gate_mean_by_class",
    "gate_histogram_10bin",
    "drug_topk_type_retention",
)


def load_diagnostics(path):
    if path == "-":
        return {}
    rows = {}
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            epoch = int(row["epoch"])
            if epoch in rows:
                raise ValueError(f"Duplicate epoch {epoch} in {path}:{line_number}")
            rows[epoch] = row
    return rows


def load_alpha_history(path):
    if path == "-":
        return {}
    values = {}
    current_epoch = None
    with Path(path).open(encoding="utf-8") as input_file:
        for line in input_file:
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1)) + 1
            alpha_match = ALPHA_RE.search(line)
            if alpha_match:
                if current_epoch is None:
                    raise ValueError(f"Alpha value appears before an epoch in {path}")
                values[current_epoch] = float(alpha_match.group(1))
    return values


def summarize_array(row, field):
    values = row.get(field)
    if values is None:
        return
    numeric = [float(value) for value in values]
    row[field] = json.dumps(numeric, separators=(",", ":"))
    if not numeric:
        return
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    row[f"{field}_mean"] = mean
    row[f"{field}_std"] = variance ** 0.5
    row[f"{field}_min"] = min(numeric)
    row[f"{field}_max"] = max(numeric)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("NAME", "DIAGNOSTICS_JSONL", "TRAIN_LOG"),
        required=True,
        help="Use '-' when a run has no diagnostics JSONL or no training log.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_rows = []
    field_order = ["experiment", "epoch"]
    for name, diagnostics_path, log_path in args.run:
        diagnostics = load_diagnostics(diagnostics_path)
        alpha_history = load_alpha_history(log_path)
        epochs = sorted(set(diagnostics) | set(alpha_history))
        if not epochs:
            raise RuntimeError(f"No diagnostics found for {name}")
        for epoch in epochs:
            row = {"experiment": name, "epoch": epoch}
            row.update(diagnostics.get(epoch, {}))
            if epoch in alpha_history:
                row["fixed_substructure_alpha"] = alpha_history[epoch]
            for field in ARRAY_FIELDS:
                summarize_array(row, field)
            for field in row:
                if field not in field_order:
                    field_order.append(field)
            output_rows.append(row)

    with Path(args.output).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=field_order, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
