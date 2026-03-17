"""Utilities for loading track definitions from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .track import Track, TrackRecords, TrackSurface, Vec2
from .track_builder import (
    DEFAULT_GRASS_FRICTION,
    DEFAULT_ROAD_FRICTION,
    SplineTrackConfig,
    build_track_from_spline,
)


@dataclass(frozen=True)
class LoadedTrack:
    """Container bundling a track with its source path."""

    name: str
    path: Path
    track: Track


class TrackLoadError(RuntimeError):
    """Raised when a track file cannot be parsed."""


def load_track(path: Path) -> Track:
    """Load a track from a JSON file."""
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:  # pragma: no cover - simple file error pass-through
        raise TrackLoadError(f"Failed to read track file {path!s}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TrackLoadError(f"Invalid JSON in track file {path!s}: {exc}") from exc

    try:
        if "surfaces" in raw:
            boundary = _parse_points(raw.get("boundary"), "boundary")
            surfaces = [_parse_surface(surface) for surface in raw.get("surfaces", [])]
            if not surfaces:
                raise TrackLoadError("Track must define at least one surface")
            spawn_point = _parse_point(raw.get("spawn_point", (0.0, 0.0)), "spawn_point")
            spawn_direction = None
            if raw.get("spawn_direction") is not None:
                spawn_direction = _parse_vector(raw.get("spawn_direction"), "spawn_direction")
            track = Track(
                boundary=boundary,
                surfaces=surfaces,
                spawn_point=spawn_point,
                spawn_direction=spawn_direction,
                records=_parse_records(raw.get("records")),
            )
            return track

        if "control_points" in raw:
            control_points = _parse_points(
                raw.get("control_points"), "control_points", max_vertices=None
            )
            widths = _parse_widths(raw.get("widths"), len(control_points))
            spawn_point = _parse_point(raw.get("spawn_point"), "spawn_point")
            config = SplineTrackConfig(
                control_points=control_points,
                widths=widths,
                spawn_point=spawn_point,
                samples_per_segment=int(raw.get("samples_per_segment", 8)),
                road_friction=float(raw.get("road_friction", DEFAULT_ROAD_FRICTION)),
                grass_friction=float(raw.get("grass_friction", DEFAULT_GRASS_FRICTION)),
                margin=float(raw.get("margin", 20.0)),
            )
            track = build_track_from_spline(config)
            records = _parse_records(raw.get("records"))
            if raw.get("spawn_direction") is not None:
                spawn_direction = _parse_vector(raw.get("spawn_direction"), "spawn_direction")
                track = Track(
                    boundary=track.boundary,
                    surfaces=track.surfaces,
                    spawn_point=track.spawn_point,
                    spawn_direction=spawn_direction,
                    control_points=track.control_points,
                    widths=track.widths,
                    sector_lines=track.sector_lines,
                    records=None,
                )
            if records is not None:
                track = replace(track, records=records)
            return track

        raise TrackLoadError("Track must define either 'surfaces' or 'control_points'")
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackLoadError(f"Malformed track data in {path!s}: {exc}") from exc


def discover_tracks(directory: Path) -> Dict[str, LoadedTrack]:
    """Return a mapping of track names to loaded tracks from a directory."""
    tracks: Dict[str, LoadedTrack] = {}
    if not directory.exists():
        return tracks
    for file in sorted(directory.glob("*.json")):
        try:
            track = load_track(file)
        except TrackLoadError:
            continue
        name = file.stem
        tracks[name] = LoadedTrack(name=name, path=file, track=track)
    return tracks


def _parse_surface(raw: dict) -> TrackSurface:
    name = raw.get("name", "surface")
    raw_polygons = raw.get("polygons")
    if raw_polygons is None:
        raw_polygon = raw.get("polygon")
        polygons = (
            _parse_points(raw_polygon, f"surface '{name}' polygon", max_vertices=16),
        )
    else:
        polygons = tuple(
            _parse_points(polygon, f"surface '{name}' polygon {index}", max_vertices=16)
            for index, polygon in enumerate(raw_polygons, start=1)
        )
    friction = float(raw.get("friction", raw.get("friction_modifier", 1.0)))
    fill_color = _parse_color(raw.get("fill_color"), default=(96, 96, 96))
    outline_color = _parse_color(raw.get("outline_color"), default=(150, 150, 150))
    return TrackSurface(
        name=name,
        polygons=polygons,
        friction_modifier=friction,
        fill_color=fill_color,
        outline_color=outline_color,
    )


def _parse_points(
    raw_points: Iterable[Iterable[float]] | None,
    label: str,
    *,
    max_vertices: int | None = 16,
) -> Sequence[Vec2]:
    if raw_points is None:
        raise ValueError(f"Missing points for {label}")
    points: List[Vec2] = []
    for point in raw_points:
        if len(point) != 2:
            raise ValueError(f"Each point in {label} must have two coordinates")
        x, y = float(point[0]), float(point[1])
        points.append((x, y))
    if len(points) < 3:
        raise ValueError(f"{label} requires at least three points")
    if max_vertices is not None and len(points) > max_vertices:
        raise ValueError(
            f"{label} has {len(points)} vertices; Box2D supports at most {max_vertices}."
        )
    return tuple(points)


def _parse_widths(raw_widths, expected: int) -> Sequence[float]:
    if raw_widths is None:
        raise ValueError("Missing widths array for spline track")
    if len(raw_widths) != expected:
        raise ValueError("Widths array must match number of control points")
    widths = [float(value) for value in raw_widths]
    if any(width <= 0 for width in widths):
        raise ValueError("All widths must be positive")
    return tuple(widths)


def _parse_point(raw_point, label: str) -> Vec2:
    if raw_point is None:
        raise ValueError(f"Missing point for {label}")
    if len(raw_point) != 2:
        raise ValueError(f"{label} must contain two values")
    return float(raw_point[0]), float(raw_point[1])


def _parse_color(raw_color, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if raw_color is None:
        return default
    if len(raw_color) != 3:
        raise ValueError("Color must contain exactly three values")
    return tuple(int(component) for component in raw_color)


def _parse_vector(raw_vec, label: str) -> Vec2:
    x, y = _parse_point(raw_vec, label)
    length = (x * x + y * y) ** 0.5
    if length == 0:
        raise ValueError(f"{label} must not be the zero vector")
    return x / length, y / length


def _parse_records(raw: dict | None) -> TrackRecords | None:
    if raw is None:
        return None
    lap_time = raw.get("fastest_lap_time")
    fastest_lap = float(lap_time) if lap_time is not None else None
    sector_times = raw.get("fastest_sector_times")
    parsed_sectors: Sequence[float | None] | None = None
    if sector_times is not None:
        parsed_sectors = tuple(
            float(value) if value is not None else None for value in sector_times
        )
    return TrackRecords(
        fastest_lap_time=fastest_lap,
        fastest_sector_times=parsed_sectors,
    )


__all__ = ["LoadedTrack", "TrackLoadError", "discover_tracks", "load_track"]
