"""Deterministic checks for validation-only checkpoint summarization."""

from pathlib import Path
import tempfile

from experiment_guard import enforce_provisional_run_limit
from select_validation_winner import decide, load_complete_summary
from summarize_validation_runs import better, parse_log, require_complete_epochs


def metric(macro_f1, kappa, acc):
    return {"Macro-F1": macro_f1, "Kappa": kappa, "ACC": acc}


def main():
    incumbent = metric(0.8000, 0.70, 0.75)
    # Strictly less than 0.001 invokes Kappa, exactly 0.001 invokes Macro-F1.
    assert better(metric(0.8009, 0.71, 0.70), incumbent)
    assert not better(metric(0.8009, 0.69, 0.90), incumbent)
    assert better(metric(0.8010, 0.60, 0.60), incumbent)
    assert not better(metric(0.7990, 0.99, 0.99), incumbent)
    assert better(metric(0.8000, 0.70, 0.76), incumbent)

    decision = decide(
        [
            {"experiment": "B", **metric(0.8009, 0.80, 0.95)},
            {"experiment": "A", **metric(0.8000, 0.90, 0.90)},
        ]
    )
    assert decision["winner"]["experiment"] == "A"
    assert decision["contenders"] == ["A", "B"]
    decision = decide(
        [
            {"experiment": "A", **metric(0.8000, 0.90, 0.90)},
            {"experiment": "C", **metric(0.8020, 0.10, 0.10)},
        ]
    )
    assert decision["winner"]["experiment"] == "C"
    assert decision["contenders"] == ["C"]
    assert decision["requires_tradeoff_review"]

    text = """epoch is 0 || Train batch_loss is 2.0
ACC:0.5, Kappa:0.4, Macro-F1:0.3
epoch is 2 || Train batch_loss is 1.0
ACC:0.7, Kappa:0.6, Macro-F1:0.5
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.log"
        path.write_text(text, encoding="utf-8")
        rows = parse_log(path)
    assert [row["epoch"] for row in rows] == [1, 3]
    assert [row["Macro-F1"] for row in rows] == [0.3, 0.5]
    try:
        require_complete_epochs("gap", "train.log", rows, 2)
    except RuntimeError as error:
        assert "Observed=[1, 3]" in str(error)
        assert "missing=2" in str(error)
        assert "unexpected=3" in str(error)
    else:
        raise AssertionError("A missing validation epoch was accepted as complete")

    incomplete_summary = """experiment,log,completed_epochs,epoch,ACC,Kappa,Macro-F1
P1,p1.log,99,50,0.9,0.8,0.7
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "summary.csv"
        path.write_text(incomplete_summary, encoding="utf-8")
        try:
            load_complete_summary(path, expected_epochs=100)
        except ValueError as error:
            assert "completed 99 epochs" in str(error)
        else:
            raise AssertionError("An incomplete validation summary was accepted")

    class GuardConfig(dict):
        __getattr__ = dict.__getitem__

    enforce_provisional_run_limit(
        GuardConfig(provisional_dependency="upstream_winner", num_epochs=5)
    )
    try:
        enforce_provisional_run_limit(
            GuardConfig(provisional_dependency="upstream_winner", num_epochs=50)
        )
    except RuntimeError as error:
        assert "upstream_winner" in str(error)
        assert "at most 5" in str(error)
    else:
        raise AssertionError("A provisional screen run was accepted")
    try:
        enforce_provisional_run_limit(
            GuardConfig(provisional_dependency="upstream_winner", num_epochs=3),
            evaluation_requested=True,
        )
    except RuntimeError as error:
        assert "checkpoint evaluation is disabled" in str(error)
    else:
        raise AssertionError("Provisional checkpoint evaluation was accepted")
    enforce_provisional_run_limit(GuardConfig(num_epochs=100))
    print("Validation selection smoke test passed")


if __name__ == "__main__":
    main()
