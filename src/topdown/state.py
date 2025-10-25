"""Lightweight snapshots of simulation state for non-graphical use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .timing import TimingState

Vec2 = Tuple[float, float]


def vec2_to_tuple(value) -> Vec2:
    """Convert a Box2D vector-like object into a Python tuple."""
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class TireState:
    """Captures per-tire motion and contact information."""

    position: Vec2
    forward_velocity: Vec2
    lateral_velocity: Vec2
    contacts: Sequence[str]


@dataclass(frozen=True)
class CarState:
    """Minimal snapshot of the car body and tire dynamics."""

    position: Vec2
    linear_velocity: Vec2
    angle: float
    angular_velocity: float
    tire_states: Sequence[TireState]


@dataclass(frozen=True)
class SimulationSnapshot:
    """High-level view of a simulation step suitable for Gym wrappers."""

    car: CarState
    timing: TimingState | None
    tire_contacts: Sequence[Sequence[str]]
    sensor_distances: Sequence[float]
    elapsed_time: float
    step_index: int


__all__ = [
    "Vec2",
    "vec2_to_tuple",
    "TireState",
    "CarState",
    "SimulationSnapshot",
]
