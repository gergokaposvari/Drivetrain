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
    raw: List[DiscreteControl] = []
    for throttle in throttle_levels:
        for steer in steer_levels:
            ctrl = DiscreteControl(throttle=throttle, steer=steer)
            if ctrl not in raw:
                raw.append(ctrl)

    center: List[DiscreteControl] = []
    forward_straight: List[DiscreteControl] = []
    forward_turns: List[DiscreteControl] = []
    coast_turns: List[DiscreteControl] = []
    reverse_turns: List[DiscreteControl] = []
    reverse_straight: List[DiscreteControl] = []

    for ctrl in raw:
        if ctrl.throttle == 0 and ctrl.steer == 0:
            center.append(ctrl)
        elif ctrl.throttle > 0 and ctrl.steer == 0:
            forward_straight.append(ctrl)
        elif ctrl.throttle > 0 and ctrl.steer != 0:
            forward_turns.append(ctrl)
        elif ctrl.throttle == 0 and ctrl.steer != 0:
            coast_turns.append(ctrl)
        elif ctrl.throttle < 0 and ctrl.steer != 0:
            reverse_turns.append(ctrl)
        elif ctrl.throttle < 0 and ctrl.steer == 0:
            reverse_straight.append(ctrl)

    return (
        center
        + forward_straight
        + sorted(forward_turns, key=lambda c: c.steer)
        + sorted(coast_turns, key=lambda c: c.steer)
        + sorted(reverse_turns, key=lambda c: c.steer)
        + reverse_straight
    )


__all__ = ["DiscreteControl", "control_to_keys", "enumerate_controls"]
