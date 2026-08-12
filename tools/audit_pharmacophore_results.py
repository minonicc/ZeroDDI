"""Audit the machine-readable pharmacophore experiment ledger."""

import csv
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results" / "pharmacophore_results.csv"
REQUIRED_COLUMNS = {
    "experiment",
    "phase",
    "run_level",
    "status",
    "metric_split",
    "seed",
    "max_epochs",
    "best_epoch",
    "ACC",
    "Kappa",
    "Macro-F1",
    "config",
    "commit",
    "physical_gpu",
    "log",
    "checkpoint",
    "change",
}
REQUIRED_RUNS = {
    ("P1", "debug"),
    ("P1", "screen50"),
    ("P1", "formal100"),
    ("P2", "debug"),
    ("P2", "screen50"),
    ("P2", "formal100"),
    ("P3", "debug"),
    ("P3", "screen50"),
    ("P3", "formal100"),
    ("Struct-S1", "debug"),
    ("Struct-S3", "debug"),
    ("Struct-S4", "engineering"),
    ("Struct-S4", "debug"),
    ("T2-64", "debug"),
    ("T3-128", "debug"),
    ("T2-256", "debug"),
    ("D3-12", "engineering"),
    ("D3-12", "debug"),
    ("D3-16", "debug"),
}


def main():
    with RESULTS.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if set(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("Unexpected pharmacophore result-table columns")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("Malformed CSV row with extra columns")

    keys = [(row["experiment"], row["run_level"], row["seed"]) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment/run-level/seed rows: {duplicates}")
    available_runs = {(experiment, run_level) for experiment, run_level, _ in keys}
    missing = sorted(REQUIRED_RUNS - available_runs)
    if missing:
        raise ValueError(f"Missing required experiment rows: {missing}")

    allowed_status = {"provided", "pending", "running", "complete", "superseded"}
    for row in rows:
        if row["status"] not in allowed_status:
            raise ValueError(f"Unsupported status in {row['experiment']}: {row['status']}")
        metrics = [row[name] for name in ("ACC", "Kappa", "Macro-F1")]
        if any(metrics) and not all(metrics):
            raise ValueError(f"Partial primary metrics in {row['experiment']}")
        if row["status"] in {"pending", "running"} and any(metrics):
            raise ValueError(f"Unfinished row has metrics in {row['experiment']}")
        if (
            row["phase"] != "historical"
            and row["run_level"] != "finaltest"
            and row["metric_split"] != "validation"
        ):
            raise ValueError(
                f"Preselection row must be validation-only: {row['experiment']}"
            )
        if row["run_level"] == "finaltest" and row["metric_split"] != "test":
            raise ValueError(f"Final-test row is not test metrics: {row['experiment']}")
        if row["run_level"] == "formal100" and row["status"] == "running":
            if not row["checkpoint"]:
                raise ValueError(f"Running formal row lacks checkpoint path: {row['experiment']}")

    print(f"Pharmacophore result-table audit passed: {len(rows)} rows")


if __name__ == "__main__":
    main()
