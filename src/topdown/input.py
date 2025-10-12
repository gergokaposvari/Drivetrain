"""Input mapping utilities for the top-down car simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import pygame


@dataclass(frozen=True)
class InputMapping:
    """Maps pygame key constants to the example key strings."""

    key_map: Dict[int, str]

    @staticmethod
    def default() -> "InputMapping":
        return InputMapping(
            {
                pygame.K_w: "up",
                pygame.K_s: "down",
                pygame.K_a: "left",
                pygame.K_d: "right",
            }
        )


class InputHandler:
    """Maintains the set of active control keys."""

    def __init__(self, mapping: InputMapping | None = None) -> None:
        self._mapping = mapping or InputMapping.default()
        self._active: Set[str] = set()

    @property
    def active(self) -> Set[str]:
        return self._active

    def process_event(self, event: pygame.event.Event) -> None:
        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key_name = self._mapping.key_map.get(event.key)
            if key_name is None:
                return
            if event.type == pygame.KEYDOWN:
                self._active.add(key_name)
            else:
                self._active.discard(key_name)

    def clear(self) -> None:
        self._active.clear()


__all__ = ["InputHandler", "InputMapping"]
