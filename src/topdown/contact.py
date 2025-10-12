"""Contact helpers for tracking tire traction."""

from __future__ import annotations

from Box2D import (  # type: ignore import-untyped
    b2Contact,
    b2ContactListener,
)

from .ground import GroundArea
from .tire import Tire


class TopDownContactListener(b2ContactListener):
    """Updates tire traction when entering or exiting ground areas."""

    def __init__(self, callback=None) -> None:
        super().__init__()
        self._callback = callback

    def _handle_contact(self, contact: b2Contact, began: bool) -> None:
        fixture_a = contact.fixtureA
        fixture_b = contact.fixtureB

        body_a = fixture_a.body
        body_b = fixture_b.body
        data_a = getattr(body_a, "userData", None)
        data_b = getattr(body_b, "userData", None)
        if not data_a or not data_b:
            return

        tire: Tire | None = None
        ground_area: GroundArea | None = None
        for data in (data_a, data_b):
            obj = data.get("obj") if isinstance(data, dict) else None
            if isinstance(obj, Tire):
                tire = obj
            elif isinstance(obj, GroundArea):
                ground_area = obj

        if tire is None or ground_area is None:
            return

        if began:
            tire.add_ground_area(ground_area)
        else:
            tire.remove_ground_area(ground_area)

        if self._callback is not None:
            self._callback(tire, ground_area, began)

    def BeginContact(self, contact: b2Contact) -> None:  # noqa: N802
        self._handle_contact(contact, True)

    def EndContact(self, contact: b2Contact) -> None:  # noqa: N802
        self._handle_contact(contact, False)


__all__ = ["TopDownContactListener"]
