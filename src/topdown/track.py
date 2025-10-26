"""Track definitions for the top-down car simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from Box2D import b2Vec2  # type: ignore import-untyped


Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class TrackSurface:
    """One or more polygons with associated traction and render colors."""

    name: str
    polygons: Sequence[Sequence[Vec2]]
    friction_modifier: float
    fill_color: Tuple[int, int, int] = (96, 96, 96)
    outline_color: Optional[Tuple[int, int, int]] = (150, 150, 150)


@dataclass(frozen=True)
class TrackRecords:
    """Stores lap record metadata for a track."""

    fastest_lap_time: Optional[float] = None
    fastest_sector_times: Optional[Sequence[Optional[float]]] = None


@dataclass(frozen=True)
class Track:
    """Describes the race surface and outer boundary."""

    boundary: Sequence[Vec2]
    surfaces: Sequence[TrackSurface]
    spawn_point: Vec2 = (0.0, 0.0)
    spawn_direction: Vec2 | None = None
    control_points: Sequence[Vec2] | None = None
    widths: Sequence[float] | None = None
    sector_lines: Sequence[Tuple[Vec2, Vec2]] | None = None
    records: TrackRecords | None = None


def default_track() -> Track:
    """Return a simple track shape with an opening straight and a few turns."""
    boundary = (
        (-60.0, -40.0),
        (-60.0, 40.0),
        (60.0, 40.0),
        (60.0, -40.0),
    )

    road_polygon = (
        (-40.0, -6.0),
        (-10.0, -6.0),
        (5.0, -12.0),
        (22.0, -12.0),
        (35.0, -4.0),
        (36.0, 8.0),
        (20.0, 16.0),
        (0.0, 18.0),
        (-18.0, 14.0),
        (-34.0, 6.0),
        (-42.0, -2.0),
    )

    pit_lane = (
        (-20.0, -12.0),
        (-5.0, -18.0),
        (15.0, -18.0),
        (28.0, -14.0),
        (26.0, -6.0),
        (8.0, -2.0),
        (-10.0, -4.0),
        (-22.0, -8.0),
    )

    grass_fill = (52, 94, 52)
    grass_outline = (70, 120, 70)
    road_fill = (96, 96, 96)
    road_outline = (150, 150, 150)

    surfaces = (
        TrackSurface(
            name="grass",
            polygons=(boundary,),
            friction_modifier=0.45,
            fill_color=grass_fill,
            outline_color=grass_outline,
        ),
        TrackSurface(
            name="road",
            polygons=(road_polygon,),
            friction_modifier=1.15,
            fill_color=road_fill,
            outline_color=road_outline,
        ),
        TrackSurface(
            name="pit",
            polygons=(pit_lane,),
            friction_modifier=0.9,
            fill_color=(110, 110, 110),
            outline_color=(180, 180, 180),
        ),
    )
    spawn_direction = (0.0, 1.0)
    return Track(
        boundary=boundary,
        surfaces=surfaces,
        spawn_point=(-35.0, -4.0),
        spawn_direction=spawn_direction,
        records=TrackRecords(),
    )


__all__ = ["Track", "TrackRecords", "TrackSurface", "default_track"]
