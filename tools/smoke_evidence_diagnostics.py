"""Deterministic checks for pharmacophore diagnostic aggregation."""

import math
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train import (
    EvidenceDiagnosticsAccumulator,
    compact_diagnostic_summary,
    save_validation_best,
)


def main():
    accumulator = EvidenceDiagnosticsAccumulator()
    # Unequal batch sizes expose accidental averaging of per-batch statistics.
    accumulator.update(
        {
            "pharmacophore_gate": torch.tensor([[[0.0], [0.2]]]),
            "pharmacophore_attention": torch.tensor([[[0.5, 0.5]]]),
            "pharmacophore_valid_pair_count": torch.tensor([2]),
        }
    )
    accumulator.update(
        {
            "pharmacophore_gate": torch.tensor(
                [
                    [[0.4], [0.6]],
                    [[0.8], [1.0]],
                    [[0.1], [0.3]],
                ]
            ),
            "pharmacophore_attention": torch.tensor(
                [[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]
            ),
            "pharmacophore_valid_pair_count": torch.tensor([4, 6, 8]),
        }
    )
    summary = accumulator.summarize()
    values = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3])
    assert math.isclose(summary["gate_mean"], float(values.mean()), abs_tol=1e-7)
    assert math.isclose(
        summary["gate_std"],
        float(values.std(unbiased=False)),
        abs_tol=1e-7,
    )
    assert math.isclose(sum(summary["gate_histogram_10bin"]), 1.0, abs_tol=1e-12)
    assert len(summary["gate_histogram_10bin_by_class"]) == 2
    assert all(len(row) == 10 for row in summary["gate_histogram_10bin_by_class"])
    assert math.isclose(summary["attention_top1_mass"], 0.875, abs_tol=1e-12)
    assert math.isclose(summary["null_token_weight"], 0.125, abs_tol=1e-12)
    assert math.isclose(summary["available_pair_count"], 5.0, abs_tol=1e-12)
    assert math.isclose(summary["valid_pair_count"], 5.0, abs_tol=1e-12)
    compact = compact_diagnostic_summary(summary)
    assert compact["gate_mean"] == summary["gate_mean"]
    assert "gate_mean_by_class" not in compact
    assert "gate_histogram_10bin_by_class" not in compact

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "best.pkl"
        cfg = SimpleNamespace(
            model_parameter_best=str(checkpoint), seednumber=42
        )
        state = {"weight": torch.tensor([1.0, 2.0])}
        metrics = {"ACC": 0.8, "Kappa": 0.7, "Macro-F1": 0.6}
        save_validation_best(state, metrics, 17, cfg)
        assert torch.equal(torch.load(checkpoint)["weight"], state["weight"])
        metadata = json.loads(
            Path(f"{checkpoint}.metrics.json").read_text(encoding="utf-8")
        )
        assert metadata["best_epoch"] == 17
        assert metadata["seed"] == 42
        assert metadata["metric_split"] == "validation"
        assert metadata["metrics"] == metrics
        assert not Path(f"{checkpoint}.metrics.json.tmp").exists()
    print("Evidence diagnostics smoke test passed")


if __name__ == "__main__":
    main()
