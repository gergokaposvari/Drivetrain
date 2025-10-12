"""Helpers for Catmull-Rom spline interpolation."""

from __future__ import annotations

from typing import List, Sequence, Tuple

Vec2 = Tuple[float, float]


def catmull_rom_spline(
    points: Sequence[Vec2],
    *,
    samples_per_segment: int,
    closed: bool = True,
) -> Tuple[List[Vec2], List[Vec2]]:
    """Return sampled positions and tangents along a Catmull-Rom spline."""
    if len(points) < 4:
        raise ValueError("Catmull-Rom spline requires at least 4 control points")
    if samples_per_segment < 1:
        raise ValueError("samples_per_segment must be >= 1")

    pts = list(points)
    if closed:
        pts = list(points) + [points[0], points[1], points[2]]
    else:
        pts = [points[0]] + list(points) + [points[-1]]

    positions: List[Vec2] = []
    tangents: List[Vec2] = []

    for i in range(1, len(pts) - 2):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2]

        for j in range(samples_per_segment):
            t = j / samples_per_segment
            pos = _catmull_rom_point(p0, p1, p2, p3, t)
            tan = _catmull_rom_tangent(p0, p1, p2, p3, t)
            positions.append(pos)
            tangents.append(_normalize(tan))

    # include end point
    positions.append(pts[-2])
    tangents.append(_normalize(_sub(pts[-2], pts[-3])))

    return positions, tangents


def _catmull_rom_point(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return x, y


def _catmull_rom_tangent(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: float) -> Vec2:
    t2 = t * t
    x = 0.5 * (
        (-p0[0] + p2[0])
        + 2 * (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t
        + 3 * (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t2
    )
    y = 0.5 * (
        (-p0[1] + p2[1])
        + 2 * (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t
        + 3 * (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t2
    )
    return x, y


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return a[0] - b[0], a[1] - b[1]


def _normalize(v: Vec2) -> Vec2:
    import math

    length = math.hypot(v[0], v[1])
    if length == 0:
        return 0.0, 0.0
    return v[0] / length, v[1] / length
