"""Check the formal/default and opt-in batch progress logging policy."""

from pathlib import Path


def main():
    source = (Path(__file__).parent / "train.py").read_text(encoding="utf-8")
    assert 'cfg.get("train_progress_interval", 0)' in source
    assert 'cfg.get("log_config", {}).get("interval", 50)' not in source
    print("Training log policy smoke test passed")


if __name__ == "__main__":
    main()
