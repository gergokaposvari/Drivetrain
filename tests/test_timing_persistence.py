import json
from pathlib import Path

import pytest

from src.topdown.timing import TimingManager
from src.topdown.track_loader import load_track, save_track_timing


def _make_basic_track(tmp_path: Path) -> Path:
    control_points = [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ]
    widths = [10.0] * len(control_points)
    track_payload = {
        "spawn_point": [0.0, 0.0],
        "control_points": control_points,
        "widths": widths,
    }
    track_path = tmp_path / "sample_track.json"
    track_path.write_text(json.dumps(track_payload))
    return track_path


def test_save_and_reload_best_lap_data(tmp_path: Path) -> None:
    track_path = _make_basic_track(tmp_path)
    track = load_track(track_path)
    assert track.best_lap_time is None
    assert track.best_lap_sector_times is None

    sector_times = [12.3, 11.2, 13.4, 10.8]
    save_track_timing(track_path, best_lap_time=47.7, best_sector_times=sector_times)

    reloaded = load_track(track_path)
    assert reloaded.best_lap_time == pytest.approx(47.7)
    assert reloaded.best_lap_sector_times is not None
    assert list(reloaded.best_lap_sector_times) == pytest.approx(sector_times)


def test_timing_manager_initialises_from_best_data() -> None:
    control_points = (
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    )
    widths = (10.0, 10.0, 10.0, 10.0)
    best_sectors = (12.3, 11.2, 13.4, 10.8)
    manager = TimingManager(
        control_points,
        widths,
        best_lap_time=47.7,
        best_sector_times=best_sectors,
    )

    state = manager.state
    assert state.best_lap_time == pytest.approx(47.7)
    assert manager.best_lap_sector_times == pytest.approx(best_sectors)


def test_timing_manager_emits_callback_on_new_best_lap() -> None:
    control_points = (
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    )
    widths = (10.0, 10.0, 10.0, 10.0)
    captured: list[tuple[float, tuple[float, ...]]] = []

    manager = TimingManager(
        control_points,
        widths,
        on_best_lap=lambda lap, sectors: captured.append((lap, tuple(sectors))),
    )

    manager._start_new_lap()  # type: ignore[attr-defined]
    manager._invalid = False  # type: ignore[attr-defined]
    manager._sector_times_current = [1.0, 1.1, 1.2, 1.3]  # type: ignore[attr-defined]
    manager._current_time = sum(manager._sector_times_current)  # type: ignore[attr-defined]
    manager._complete_lap()  # type: ignore[attr-defined]

    assert len(captured) == 1
    lap_time, sectors = captured[0]
    assert lap_time == pytest.approx(4.6)
    assert sectors == pytest.approx((1.0, 1.1, 1.2, 1.3))
    assert manager.best_lap_sector_times == pytest.approx((1.0, 1.1, 1.2, 1.3))
