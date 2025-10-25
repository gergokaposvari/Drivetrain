import json
from pathlib import Path

from exec import TrainerProgress, load_progress, save_progress


def test_trainer_progress_persistence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "progress.json"
    progress = TrainerProgress()
    progress.record(10.0)
    progress.record(5.0)
    save_progress(checkpoint, progress)

    loaded = load_progress(checkpoint)
    assert loaded.episodes_completed == 2
    assert loaded.best_reward == 10.0
    assert loaded.reward_history[-1] == 5.0


def test_load_nonexistent_checkpoint_returns_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing.json"
    progress = load_progress(checkpoint)
    assert progress.episodes_completed == 0
    assert progress.best_reward is None
