"""Raycast-style sensor helpers implemented with lightweight geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Vec2 = Tuple[float, float]
Segment = Tuple[Vec2, Vec2]


@dataclass(frozen=True)
class RaycastConfig:
    """Configuration for the raycast sensor suite."""

    angles_deg: Sequence[float] = (-60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0)
    max_distance: float = 150.0


@dataclass(frozen=True)
class RaycastSample:
    """Sensor output capturing the hit distances for each ray."""

    distances: Tuple[float, ...]
    hits: Tuple[bool, ...]

    def normalized(self, max_distance: float) -> Tuple[float, ...]:
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        return tuple(distance / max_distance for distance in self.distances)


class RaycastSensors:
    """Computes distances from the car to the road boundary using line segments."""

    def __init__(
        self,
        segments: Iterable[Segment] | None = None,
        config: RaycastConfig | None = None,
    ) -> None:
        self._config = config or RaycastConfig()
        self._segments: List[Segment] = []
        if segments is not None:
            self.set_segments(segments)

    @property
    def angles(self) -> Tuple[float, ...]:
        return tuple(self._config.angles_deg)

    @property
    def max_distance(self) -> float:
        return self._config.max_distance

    def set_segments(self, segments: Iterable[Segment]) -> None:
        self._segments = [(_to_vec(a), _to_vec(b)) for a, b in segments]

    def sample(self, origin: Vec2, forward: Vec2) -> RaycastSample:
        distances: List[float] = []
        hits: List[bool] = []

        forward = _normalize(forward, default=(0.0, 1.0))
        right = (-forward[1], forward[0])

        for angle_deg in self._config.angles_deg:
            theta = math.radians(angle_deg)
            direction = (
                math.cos(theta) * forward[0] + math.sin(theta) * right[0],
                math.cos(theta) * forward[1] + math.sin(theta) * right[1],
            )
            direction = _normalize(direction, default=forward)
            distance = self._cast_ray(origin, direction)
            if distance is None or distance > self._config.max_distance:
                distances.append(self._config.max_distance)
                hits.append(False)
            else:
                distances.append(distance)
                hits.append(True)

        return RaycastSample(distances=tuple(distances), hits=tuple(hits))

    def _cast_ray(self, origin: Vec2, direction: Vec2) -> float | None:
        best_distance: float | None = None
        for segment in self._segments:
            distance = _ray_segment_distance(origin, direction, segment)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
        return best_distance


def _to_vec(vec: Vec2) -> Vec2:
    return float(vec[0]), float(vec[1])


def _normalize(vec: Vec2, *, default: Vec2) -> Vec2:
    length = math.hypot(vec[0], vec[1])
    if length == 0:
        return default
    return vec[0] / length, vec[1] / length


def _ray_segment_distance(origin: Vec2, direction: Vec2, segment: Segment) -> float | None:
    px, py = origin
    dx, dy = direction
    (ax, ay), (bx, by) = segment
    sx = bx - ax
    sy = by - ay

    denominator = dx * sy - dy * sx
    if abs(denominator) < 1e-8:
        return None

    diff_x = ax - px
    diff_y = ay - py
    t = (diff_x * sy - diff_y * sx) / denominator
    if t < 0:
        return None

    u = (diff_x * dy - diff_y * dx) / denominator
    if u < -1e-6 or u > 1.0 + 1e-6:
        return None

    return t


__all__ = ["RaycastConfig", "RaycastSample", "RaycastSensors", "Segment", "Vec2"]
