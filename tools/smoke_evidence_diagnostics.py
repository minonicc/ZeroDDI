"""Deterministic checks for pharmacophore diagnostic aggregation."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train import EvidenceDiagnosticsAccumulator


def main():
    accumulator = EvidenceDiagnosticsAccumulator()
    # Unequal batch sizes expose accidental averaging of per-batch statistics.
    accumulator.update(
        {"pharmacophore_gate": torch.tensor([[[0.0], [0.2]]])}
    )
    accumulator.update(
        {
            "pharmacophore_gate": torch.tensor(
                [
                    [[0.4], [0.6]],
                    [[0.8], [1.0]],
                    [[0.1], [0.3]],
                ]
            )
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
    print("Evidence diagnostics smoke test passed")


if __name__ == "__main__":
    main()
