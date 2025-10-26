"""Lap timing utilities for spline-based tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class SectorStatus:
    completed: bool
    faster: bool | None
    current_time: float | None
    best_time: float | None


@dataclass(frozen=True)
class TimingState:
    running: bool
    invalid: bool
    current_time: float
    sector_statuses: Sequence[SectorStatus]
    best_lap_time: float | None
    last_lap_time: float | None


@dataclass(frozen=True)
class LapResult:
    """Outcome of a completed lap."""

    lap_time: float | None
    sector_times: Sequence[float | None]
    sector_improved: Sequence[bool]
    invalid: bool
    new_best_lap: bool


class TimingManager:
    """Tracks mini-sector timings along the track control points."""

    def __init__(
        self,
        control_points: Sequence[Vec2],
        widths: Sequence[float],
        sector_lines: Sequence[Tuple[Vec2, Vec2]] | None = None,
        *,
        lap_callback: Callable[[LapResult], None] | None = None,
        best_lap_time: float | None = None,
        best_sector_times: Sequence[float | None] | None = None,
    ) -> None:
        if len(control_points) < 4:
            raise ValueError("TimingManager requires at least four control points")
        if len(control_points) != len(widths):
            raise ValueError("Control points and widths must match in length")

        self._points = list(control_points)
        self._widths = list(widths)
        self._num_points = len(self._points)

        if sector_lines is not None:
            if len(sector_lines) != self._num_points:
                raise ValueError("Sector lines must match number of control points")
            self._segments = list(sector_lines)
        else:
            self._segments = [
                (self._points[i], self._points[(i + 1) % self._num_points])
                for i in range(self._num_points)
            ]

        self._best_sector_times: List[float | None] = (
            list(best_sector_times) if best_sector_times is not None else [None] * self._num_points
        )
        if len(self._best_sector_times) != self._num_points:
            raise ValueError("best_sector_times must match number of control points")
        self._best_lap_time: float | None = best_lap_time
        self._last_lap_time: float | None = None
        self._lap_callback = lap_callback

        self._running = False
        self._awaiting_start_input = True
        self._invalid = False
        self._on_road = True

        self._current_time = 0.0
        self._current_segment_index = 0
        self._last_sector_timestamp = 0.0
        self._sector_times_current: List[float | None] = [None] * self._num_points
        self._sector_improved_current: List[bool] = [False] * self._num_points
        self._sector_statuses: List[SectorStatus] = [
            SectorStatus(completed=False, faster=None, current_time=None, best_time=None)
            for _ in range(self._num_points)
        ]

        self._prev_position: Vec2 | None = None

    def set_on_road(self, on_road: bool) -> None:
        self._on_road = on_road
        if self._running and not on_road:
            self._invalid = True

    def update(self, dt: float, has_input: bool, car_position: Vec2) -> None:
        if self._awaiting_start_input:
            if has_input:
                self._start_new_lap(initial=True)
            else:
                return

        if not self._running:
            return

        self._current_time += dt

        prev = self._prev_position
        self._prev_position = car_position

        if not self._on_road:
            self._invalid = True

        if prev is None:
            return

        current_index = self._current_segment_index
        if self._segment_crossed(prev, car_position, self._segments[current_index]):
            self._handle_sector_reached(current_index)

    def _start_new_lap(self, initial: bool = False) -> None:
        self._awaiting_start_input = False
        self._running = True
        self._invalid = False
        self._current_time = 0.0
        self._last_sector_timestamp = 0.0
        self._sector_times_current = [None] * self._num_points
        self._sector_improved_current = [False] * self._num_points
        self._sector_statuses = [
            SectorStatus(
                completed=False,
                faster=None,
                current_time=None,
                best_time=self._best_sector_times[i],
            )
            for i in range(self._num_points)
        ]
        self._current_segment_index = 0
        self._prev_position = None

    def _segment_crossed(self, start: Vec2, end: Vec2, segment: Tuple[Vec2, Vec2]) -> bool:
        seg_start, seg_end = segment
        return _segments_intersect(start, end, seg_start, seg_end)

    def _handle_sector_reached(self, segment_index: int) -> None:
        if self._num_points == 0:
            return
        sector_index = segment_index
        sector_time = self._current_time - self._last_sector_timestamp
        self._sector_times_current[sector_index] = sector_time

        best_time = self._best_sector_times[sector_index]
        faster = None
        if best_time is None:
            faster = True
            self._best_sector_times[sector_index] = sector_time
            self._sector_improved_current[sector_index] = True
        else:
            faster = sector_time < best_time
            if faster:
                self._best_sector_times[sector_index] = sector_time
            self._sector_improved_current[sector_index] = bool(faster)

        self._sector_statuses[sector_index] = SectorStatus(
            completed=True,
            faster=faster,
            current_time=sector_time,
            best_time=self._best_sector_times[sector_index],
        )

        self._last_sector_timestamp = self._current_time
        self._current_segment_index = (self._current_segment_index + 1) % self._num_points

        if self._current_segment_index == 0:
            self._complete_lap()
            self._current_segment_index = 0
            self._prev_position = None

    def _complete_lap(self) -> None:
        sector_times_snapshot = tuple(self._sector_times_current)
        improved_snapshot = tuple(self._sector_improved_current)
        lap_valid = not self._invalid and all(time is not None for time in sector_times_snapshot)
        lap_time = self._current_time if lap_valid else None
        new_best_lap = False
        if lap_valid:
            self._last_lap_time = lap_time
            if lap_time is not None and (
                self._best_lap_time is None or lap_time < self._best_lap_time
            ):
                self._best_lap_time = lap_time
                new_best_lap = True
        else:
            self._last_lap_time = None

        self._current_time = 0.0
        self._last_sector_timestamp = 0.0
        self._invalid = False
        self._sector_times_current = [None] * self._num_points
        self._sector_improved_current = [False] * self._num_points
        self._sector_statuses = [
            SectorStatus(
                completed=False,
                faster=None,
                current_time=None,
                best_time=self._best_sector_times[i],
            )
            for i in range(self._num_points)
        ]
        if self._lap_callback is not None:
            result = LapResult(
                lap_time=lap_time,
                sector_times=sector_times_snapshot,
                sector_improved=improved_snapshot,
                invalid=not lap_valid,
                new_best_lap=new_best_lap,
            )
            self._lap_callback(result)

    @property
    def state(self) -> TimingState:
        return TimingState(
            running=self._running,
            invalid=self._invalid,
            current_time=self._current_time,
            sector_statuses=tuple(self._sector_statuses),
            best_lap_time=self._best_lap_time,
            last_lap_time=self._last_lap_time,
        )

    @property
    def segments(self) -> Sequence[Tuple[Vec2, Vec2]]:
        return tuple(self._segments)

    @property
    def best_lap_time(self) -> float | None:
        return self._best_lap_time

    @property
    def best_sector_times(self) -> Sequence[float | None]:
        return tuple(self._best_sector_times)

    def reset_run(self) -> None:
        """Reset the running state while preserving best records."""
        self._running = False
        self._awaiting_start_input = True
        self._invalid = False
        self._on_road = True
        self._current_time = 0.0
        self._current_segment_index = 0
        self._last_sector_timestamp = 0.0
        self._sector_times_current = [None] * self._num_points
        self._sector_improved_current = [False] * self._num_points
        self._sector_statuses = [
            SectorStatus(
                completed=False,
                faster=None,
                current_time=None,
                best_time=self._best_sector_times[i],
            )
            for i in range(self._num_points)
        ]
        self._prev_position = None


def _segments_intersect(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool:
    if p1 == p2:
        return False

    def orient(a: Vec2, b: Vec2, c: Vec2) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        return _overlap(p1, p2, q1, q2)

    return o1 * o2 <= 0 and o3 * o4 <= 0


def _overlap(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool:
    def in_range(a: float, b: float, c: float) -> bool:
        return min(a, b) <= c <= max(a, b)

    return (
        in_range(p1[0], p2[0], q1[0])
        or in_range(p1[0], p2[0], q2[0])
        or in_range(q1[0], q2[0], p1[0])
        or in_range(q1[0], q2[0], p2[0])
    ) and (
        in_range(p1[1], p2[1], q1[1])
        or in_range(p1[1], p2[1], q2[1])
        or in_range(q1[1], q2[1], p1[1])
        or in_range(q1[1], q2[1], p2[1])
    )
__all__ = ["TimingManager", "TimingState", "SectorStatus"]
