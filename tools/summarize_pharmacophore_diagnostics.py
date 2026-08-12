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
    "pair_topk_type_retention",
    "selected_position_histogram_64_128_256_512",
    "drug_topk_type_retention",
)
NESTED_ARRAY_FIELDS = ("gate_histogram_10bin_by_class",)
PHARMACOPHORE_FAMILIES = (
    "Hydrophobe",
    "Aromatic",
    "Donor",
    "Acceptor",
    "PosIonizable",
    "NegIonizable",
)
ARRAY_LABELS = {
    "gate_histogram_10bin": [f"[{i / 10:.1f},{(i + 1) / 10:.1f})" for i in range(10)],
    "pair_topk_type_retention": [
        f"{left}|{right}"
        for left in PHARMACOPHORE_FAMILIES
        for right in PHARMACOPHORE_FAMILIES
    ],
    "selected_position_histogram_64_128_256_512": [
        "[0,64)",
        "[64,128)",
        "[128,256)",
        "[256,512)",
        "[512,+inf)",
    ],
    "drug_topk_type_retention": list(PHARMACOPHORE_FAMILIES),
}


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
    labels = ARRAY_LABELS.get(field)
    if labels is not None:
        if len(numeric) != len(labels):
            raise ValueError(
                f"{field} has {len(numeric)} values; expected {len(labels)}"
            )
        row[f"{field}_labeled"] = json.dumps(
            dict(zip(labels, numeric)), separators=(",", ":")
        )
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
    parser.add_argument(
        "--semantics",
        nargs=2,
        action="append",
        metavar=("NAME", "AGGREGATION"),
        required=True,
        help=(
            "Declare each run as 'legacy_batch_mean' or 'element_weighted'; "
            "mixed semantics remain visible in the output."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    semantics = {}
    for name, aggregation in args.semantics:
        if name in semantics:
            raise ValueError(f"Duplicate diagnostic semantics for {name}")
        if aggregation not in {"legacy_batch_mean", "element_weighted"}:
            raise ValueError(
                f"Unsupported aggregation semantics for {name}: {aggregation}"
            )
        semantics[name] = aggregation
    run_names = [name for name, _, _ in args.run]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Diagnostic run names must be unique")
    missing_semantics = sorted(set(run_names) - set(semantics))
    extra_semantics = sorted(set(semantics) - set(run_names))
    if missing_semantics or extra_semantics:
        raise ValueError(
            f"Diagnostic semantics mismatch: missing={missing_semantics}, "
            f"extra={extra_semantics}"
        )

    output_rows = []
    field_order = ["experiment", "aggregation_semantics", "epoch"]
    for name, diagnostics_path, log_path in args.run:
        diagnostics = load_diagnostics(diagnostics_path)
        alpha_history = load_alpha_history(log_path)
        epochs = sorted(set(diagnostics) | set(alpha_history))
        if not epochs:
            raise RuntimeError(f"No diagnostics found for {name}")
        for epoch in epochs:
            row = {
                "experiment": name,
                "aggregation_semantics": semantics[name],
                "epoch": epoch,
            }
            row.update(diagnostics.get(epoch, {}))
            if epoch in alpha_history:
                row["fixed_substructure_alpha"] = alpha_history[epoch]
            for field in ARRAY_FIELDS:
                summarize_array(row, field)
            for field in NESTED_ARRAY_FIELDS:
                if field in row:
                    row[field] = json.dumps(row[field], separators=(",", ":"))
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
