"""Helpers for expressing control inputs without relying on pygame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set


@dataclass(frozen=True)
class DiscreteControl:
    """Simple throttle/steer control abstraction."""

    throttle: int  # -1 = brake, 0 = coasting, 1 = accelerate
    steer: int  # -1 = left, 0 = straight, 1 = right

    def clamp(self) -> "DiscreteControl":
        return DiscreteControl(
            throttle=max(-1, min(1, self.throttle)),
            steer=max(-1, min(1, self.steer)),
        )


def control_to_keys(control: DiscreteControl) -> Set[str]:
    """Convert a discrete control state to the string keys used by Car.update."""
    control = control.clamp()
    keys: Set[str] = set()
    if control.throttle > 0:
        keys.add("up")
    elif control.throttle < 0:
        keys.add("down")
    if control.steer > 0:
        keys.add("right")
    elif control.steer < 0:
        keys.add("left")
    return keys


def enumerate_controls(
    throttle_levels: Sequence[int] = (-1, 0, 1),
    steer_levels: Sequence[int] = (-1, 0, 1),
) -> List[DiscreteControl]:
    """Return all combinations of the supplied throttle and steer levels."""
    controls: List[DiscreteControl] = []
    for throttle in throttle_levels:
        for steer in steer_levels:
            controls.append(DiscreteControl(throttle=throttle, steer=steer))
    return controls


__all__ = ["DiscreteControl", "control_to_keys", "enumerate_controls"]
