"""Constructs track surfaces from spline control points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .spline import catmull_rom_spline
from .track import Track, TrackSurface, Vec2

MAX_B2_VERTICES = 16
DEFAULT_ROAD_FRICTION = 1.2
DEFAULT_ROAD_COLOR = (100, 100, 100)
DEFAULT_GRASS_FRICTION = 0.45
DEFAULT_GRASS_COLOR = (48, 88, 48)
DEFAULT_GRASS_OUTLINE = (70, 120, 70)


@dataclass(frozen=True)
class SplineTrackConfig:
    control_points: Sequence[Vec2]
    widths: Sequence[float]
    spawn_point: Vec2
    samples_per_segment: int = 8
    road_friction: float = DEFAULT_ROAD_FRICTION
    grass_friction: float = DEFAULT_GRASS_FRICTION
    margin: float = 20.0


def build_track_from_spline(config: SplineTrackConfig) -> Track:
    if len(config.control_points) < 4:
        raise ValueError("At least four control points are required for a closed spline")
    if len(config.control_points) != len(config.widths):
        raise ValueError("control_points and widths must have the same length")
    if any(width <= 0 for width in config.widths):
        raise ValueError("All widths must be positive")

    positions, tangents = catmull_rom_spline(
        config.control_points,
        samples_per_segment=config.samples_per_segment,
        closed=True,
    )
    width_samples = _catmull_rom_scalar(
        config.widths,
        samples_per_segment=config.samples_per_segment,
        closed=True,
    )

    left_edge: List[Vec2] = []
    right_edge: List[Vec2] = []
    control_left: List[Vec2] = []
    control_right: List[Vec2] = []
    last_normal = (0.0, 1.0)
    for idx, (pos, tan, width) in enumerate(zip(positions, tangents, width_samples)):
        normal = (-tan[1], tan[0])
        length = (normal[0] ** 2 + normal[1] ** 2) ** 0.5
        if length == 0.0:
            normal = last_normal
        else:
            normal = (normal[0] / length, normal[1] / length)
            last_normal = normal
        half = width / 2.0
        left_point = (pos[0] + normal[0] * half, pos[1] + normal[1] * half)
        right_point = (pos[0] - normal[0] * half, pos[1] - normal[1] * half)
        left_edge.append(left_point)
        right_edge.append(right_point)

        if idx % config.samples_per_segment == 0:
            control_left.append(left_point)
            control_right.append(right_point)

    # Remove duplicated wrap-around sample
    if _almost_equal(left_edge[0], left_edge[-1]) and _almost_equal(right_edge[0], right_edge[-1]):
        left_edge.pop()
        right_edge.pop()
        if len(control_left) > len(config.control_points):
            control_left.pop()
            control_right.pop()

    if len(control_left) > len(config.control_points):
        control_left = control_left[: len(config.control_points)]
        control_right = control_right[: len(config.control_points)]

    road_polygons = _build_road_polygons(left_edge, right_edge)

    boundary = _compute_boundary(left_edge + right_edge, margin=config.margin)
    sector_lines = tuple(zip(control_left, control_right))

    spawn_direction = _normalize_vec(
        (
            config.control_points[1][0] - config.control_points[0][0],
            config.control_points[1][1] - config.control_points[0][1],
        )
    )

    road_surface = TrackSurface(
        name="road",
        polygons=tuple(tuple(p for p in poly) for poly in road_polygons),
        friction_modifier=config.road_friction,
        fill_color=DEFAULT_ROAD_COLOR,
        outline_color=None,
    )
    grass_polygon = tuple(boundary)
    grass_surface = TrackSurface(
        name="grass",
        polygons=(grass_polygon,),
        friction_modifier=config.grass_friction,
        fill_color=DEFAULT_GRASS_COLOR,
        outline_color=DEFAULT_GRASS_OUTLINE,
    )

    return Track(
        boundary=grass_polygon,
        surfaces=(grass_surface, road_surface),
        spawn_point=config.spawn_point,
        spawn_direction=spawn_direction,
        control_points=tuple(config.control_points),
        widths=tuple(config.widths),
        sector_lines=sector_lines,
    )


def _catmull_rom_scalar(
    values: Sequence[float],
    *,
    samples_per_segment: int,
    closed: bool,
) -> List[float]:
    if len(values) < 4:
        raise ValueError("Scalar Catmull-Rom requires at least four values")
    pts = list(values)
    if closed:
        pts = list(values) + [values[0], values[1], values[2]]
    else:
        pts = [values[0]] + list(values) + [values[-1]]

    samples: List[float] = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            samples.append(_catmull_point_scalar(p0, p1, p2, p3, t))
    samples.append(pts[-2])
    return samples


def _catmull_point_scalar(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def _build_road_polygons(left: List[Vec2], right: List[Vec2]) -> List[List[Vec2]]:
    if len(left) != len(right):
        raise ValueError("Left and right edge lists must have the same length")
    if len(left) < 2:
        return []

    polygons: List[List[Vec2]] = []
    count = len(left)
    for i in range(count):
        j = (i + 1) % count
        polygon = [left[i], left[j], right[j], right[i]]
        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons


def _compute_boundary(points: Sequence[Vec2], *, margin: float) -> Tuple[Vec2, Vec2, Vec2, Vec2]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin
    return (
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
    )


def _almost_equal(a: Vec2, b: Vec2, tol: float = 1e-4) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _normalize_vec(vec: Vec2) -> Vec2:
    import math

    length = math.hypot(vec[0], vec[1])
    if length == 0:
        return 0.0, 1.0
    return vec[0] / length, vec[1] / length

__all__ = [
    "SplineTrackConfig",
    "build_track_from_spline",
    "DEFAULT_ROAD_FRICTION",
    "DEFAULT_GRASS_FRICTION",
]
