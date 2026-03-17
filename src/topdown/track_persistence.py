"""Utilities for persisting lap records to track JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


def update_fastest_lap(
    path: Path,
    lap_time: float,
    sector_times: Sequence[float | None],
) -> None:
    """Persist the supplied lap information into the track JSON file."""
    raw = json.loads(path.read_text())
    records = raw.get("records", {})
    records["fastest_lap_time"] = lap_time
    records["fastest_sector_times"] = [
        float(value) if value is not None else None for value in sector_times
    ]
    raw["records"] = records

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2))
    temporary.replace(path)


__all__ = ["update_fastest_lap"]
