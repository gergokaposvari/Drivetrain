import json
from pathlib import Path

import pytest

from src.topdown.track_loader import load_track
from src.topdown.track_persistence import update_fastest_lap


def _write_basic_track(path: Path) -> None:
    data = {
        "boundary": [
            [-20.0, -20.0],
            [-20.0, 20.0],
            [20.0, 20.0],
            [20.0, -20.0],
        ],
        "surfaces": [
            {
                "name": "grass",
                "polygon": [
                    [-20.0, -20.0],
                    [-20.0, 20.0],
                    [20.0, 20.0],
                    [20.0, -20.0],
                ],
                "friction": 0.5,
            },
            {
                "name": "road",
                "polygon": [
                    [-10.0, -5.0],
                    [10.0, -5.0],
                    [10.0, 5.0],
                    [-10.0, 5.0],
                ],
                "friction": 1.0,
            },
        ],
        "spawn_point": [0.0, 0.0],
        "spawn_direction": [0.0, 1.0],
    }
    path.write_text(json.dumps(data, indent=2))


def test_update_fastest_lap_round_trip(tmp_path: Path) -> None:
    track_path = tmp_path / "test_track.json"
    _write_basic_track(track_path)

    update_fastest_lap(track_path, 12.345, [3.0, 4.0, 5.0])

    raw = json.loads(track_path.read_text())
    assert raw["records"]["fastest_lap_time"] == pytest.approx(12.345)
    assert raw["records"]["fastest_sector_times"] == [3.0, 4.0, 5.0]

    loaded = load_track(track_path)
    assert loaded.records is not None
    assert loaded.records.fastest_lap_time == pytest.approx(12.345)
    assert loaded.records.fastest_sector_times == (3.0, 4.0, 5.0)
