"""Tire implementation mirroring the original pybox2d example."""

from __future__ import annotations

from typing import List, Set

from Box2D import (  # type: ignore import-untyped
    b2Body,
    b2World,
)

from .ground import GroundArea


class Tire:
    """Standalone version of the example TDTire class."""

    def __init__(
        self,
        car_body: b2Body,
        *,
        max_forward_speed: float = 100.0,
        max_backward_speed: float = -20.0,
        max_drive_force: float = 150.0,
        turn_torque: float = 15.0,
        max_lateral_impulse: float = 3.0,
        dimensions: tuple[float, float] = (0.5, 1.25),
        density: float = 1.0,
        position: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        world: b2World = car_body.world
        self.max_forward_speed = max_forward_speed
        self.max_backward_speed = max_backward_speed
        self.max_drive_force = max_drive_force
        self.turn_torque = turn_torque
        self.max_lateral_impulse = max_lateral_impulse
        self.ground_areas: List[GroundArea] = []
        self.current_traction = 1.0

        self.body: b2Body = world.CreateDynamicBody(position=position)
        self.body.CreatePolygonFixture(box=dimensions, density=density)
        self.body.userData = {"obj": self}

    @property
    def forward_velocity(self):  # type: ignore[override]
        body = self.body
        current_normal = body.GetWorldVector((0.0, 1.0))
        return current_normal.dot(body.linearVelocity) * current_normal

    @property
    def lateral_velocity(self):  # type: ignore[override]
        body = self.body
        right_normal = body.GetWorldVector((1.0, 0.0))
        return right_normal.dot(body.linearVelocity) * right_normal

    def update_friction(self) -> None:
        impulse = -self.lateral_velocity * self.body.mass
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length
        self.body.ApplyLinearImpulse(self.current_traction * impulse, self.body.worldCenter, True)

        angular_impulse = (
            0.1 * self.current_traction * self.body.inertia * -self.body.angularVelocity
        )
        self.body.ApplyAngularImpulse(angular_impulse, True)

        current_forward_normal = self.forward_velocity
        current_forward_speed = current_forward_normal.Normalize()
        drag_force_magnitude = -2.0 * current_forward_speed
        self.body.ApplyForce(
            self.current_traction * drag_force_magnitude * current_forward_normal,
            self.body.worldCenter,
            True,
        )

    def update_drive(self, keys: Set[str]) -> None:
        if "up" in keys:
            desired_speed = self.max_forward_speed
        elif "down" in keys:
            desired_speed = self.max_backward_speed
        else:
            return

        current_forward_normal = self.body.GetWorldVector((0.0, 1.0))
        current_speed = self.forward_velocity.dot(current_forward_normal)

        force = 0.0
        if desired_speed > current_speed:
            force = self.max_drive_force
        elif desired_speed < current_speed:
            force = -self.max_drive_force
        else:
            return

        self.body.ApplyForce(
            self.current_traction * force * current_forward_normal,
            self.body.worldCenter,
            True,
        )

    def update_turn(self, keys: Set[str]) -> None:
        if "left" in keys:
            desired_torque = self.turn_torque
        elif "right" in keys:
            desired_torque = -self.turn_torque
        else:
            return
        self.body.ApplyTorque(desired_torque, True)

    def add_ground_area(self, area: GroundArea) -> None:
        if area not in self.ground_areas:
            self.ground_areas.append(area)
            self.update_traction()

    def remove_ground_area(self, area: GroundArea) -> None:
        if area in self.ground_areas:
            self.ground_areas.remove(area)
            self.update_traction()

    def update_traction(self) -> None:
        if not self.ground_areas:
            self.current_traction = 1.0
            return
        self.current_traction = 0.0
        modifiers = [area.friction_modifier for area in self.ground_areas]
        max_modifier = max(modifiers)
        if max_modifier > self.current_traction:
            self.current_traction = max_modifier


__all__ = ["Tire"]
