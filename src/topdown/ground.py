"""Ground areas for the top-down car example."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundArea:
    """Area on the ground affecting tire traction."""

    name: str
    friction_modifier: float


__all__ = ["GroundArea"]
