"""Check that validation accumulates only predictions and labels linearly."""

import torch
from torch.utils.data import Dataset

from .train import evaluate


class EvaluationDataset(Dataset):
    mode = "val"
    current_dataset_eventid_uni = (0, 1, 2)
    embid2eventid = {0: 0, 1: 1, 2: 2}

    def __len__(self):
        return 130

    def __getitem__(self, index):
        return index


class UnusedOutput:
    def detach(self):
        raise AssertionError("An unused instance/prototype output was retained")


class EvaluationModel:
    def eval(self):
        return self

    def __call__(self, batch):
        labels = batch % 3
        logits = torch.full((batch.shape[0], 3), -1.0)
        logits.scatter_(1, labels[:, None], 1.0)
        return torch.tensor(0.25), logits, labels, UnusedOutput(), UnusedOutput()


class Logger:
    def info(self, *_args, **_kwargs):
        pass


def main():
    metrics = evaluate(
        EvaluationModel(),
        EvaluationDataset(),
        Logger(),
        {"selection_metric": "Macro-F1"},
        "seen",
        return_metrics=True,
    )
    assert metrics["ACC"] == 1.0
    assert metrics["Kappa"] == 1.0
    assert metrics["Macro-F1"] == 1.0
    print("Evaluation accumulation smoke test passed")


if __name__ == "__main__":
    main()
