"""Car container corresponding to the TDCar example class."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from Box2D import (  # type: ignore import-untyped
    b2Body,
    b2RevoluteJoint,
    b2Vec2,
    b2World,
)

from .tire import Tire


class Car:
    """Refactored TDCar with unchanged logic."""

    vertices = (
        (1.5, 0.0),
        (3.0, 2.5),
        (2.8, 5.5),
        (1.0, 10.0),
        (-1.0, 10.0),
        (-2.8, 5.5),
        (-3.0, 2.5),
        (-1.5, 0.0),
    )
    tire_anchors = (
        (-3.0, 0.75),
        (3.0, 0.75),
        (-3.0, 8.5),
        (3.0, 8.5),
    )

    def __init__(
        self,
        world: b2World,
        *,
        vertices: Sequence[tuple[float, float]] | None = None,
        tire_anchors: Sequence[tuple[float, float]] | None = None,
        density: float = 0.1,
        position: tuple[float, float] = (0.0, 0.0),
        angle: float = 0.0,
        **tire_kwargs,
    ) -> None:
        if vertices is None:
            vertices = Car.vertices
        if tire_anchors is None:
            tire_anchors = Car.tire_anchors

        self.vertices = tuple(vertices)
        self.tire_anchors = tuple(tire_anchors)

        self.throttle_input: float = 0.0
        self.steering_input: float = 0.0

        self.body: b2Body = world.CreateDynamicBody(position=position, angle=angle)
        self.body.CreatePolygonFixture(vertices=self.vertices, density=density)
        self.body.userData = {"obj": self}

        self._crashed = False
        self.tires = [Tire(self.body, car=self, **tire_kwargs) for _ in range(4)]
        self.joints: list[b2RevoluteJoint] = []

        for tire, anchor in zip(self.tires, self.tire_anchors):
            anchor_vec = b2Vec2(*anchor)
            world_anchor = self.body.GetWorldPoint(anchor_vec)
            tire.body.position = world_anchor
            tire.body.angle = self.body.angle

            joint = world.CreateRevoluteJoint(
                bodyA=self.body,
                bodyB=tire.body,
                localAnchorA=anchor,
                localAnchorB=(0.0, 0.0),
                enableMotor=False,
                maxMotorTorque=1000.0,
                enableLimit=True,
                lowerAngle=0.0,
                upperAngle=0.0,
            )
            assert isinstance(joint, b2RevoluteJoint)
            self.joints.append(joint)

    def update(self, keys: set[str] | None, hz: float) -> None:
        if keys is not None:
            self.throttle_input = float("up" in keys) - float("down" in keys)
            self.steering_input = float("right" in keys) - float("left" in keys)

        self._apply_continuous_controls(hz)

    def reset(self, position: tuple[float, float], angle: float) -> None:
        """Teleport the car to a specific pose and clear residual motion."""
        pos_vec = b2Vec2(*position)
        self.body.position = pos_vec
        self.body.angle = angle
        self.body.linearVelocity = b2Vec2(0.0, 0.0)
        self.body.angularVelocity = 0.0
        self.body.awake = True

        for tire, anchor, joint in zip(self.tires, self.tire_anchors, self.joints):
            anchor_vec = b2Vec2(*anchor)
            world_anchor = self.body.GetWorldPoint(anchor_vec)
            tire.body.position = world_anchor
            tire.body.angle = angle
            tire.body.linearVelocity = b2Vec2(0.0, 0.0)
            tire.body.angularVelocity = 0.0
            tire.body.awake = True
            joint.SetLimits(0.0, 0.0)

        self._set_front_wheel_angle(0.0)
        self._crashed = False

    @property
    def forward_speed(self) -> float:
        return np.dot(self.body.linearVelocity, self.forward_vector)

    @property
    def forward_vector(self) -> b2Vec2:
        return self.body.GetWorldVector((0.0, 1.0))

    @property
    def front_wheel_angle(self) -> float:
        if len(self.joints) < 4:
            return 0.0
        left_angle = self.joints[2].angle
        right_angle = self.joints[3].angle
        return 0.5 * (left_angle + right_angle)

    def _set_front_wheel_angle(self, angle: float) -> None:
        if len(self.joints) < 4:
            return
        for joint in self.joints[2:4]:
            joint.SetLimits(angle, angle)

    @property
    def crashed(self) -> bool:
        return self._crashed

    def mark_crashed(self) -> None:
        self._crashed = True

    def clear_crash(self) -> None:
        self._crashed = False

    def _apply_continuous_controls(self, hz: float) -> None:
        for tire in self.tires:
            tire.update_friction()

        lock_angle = math.radians(40.0)
        target_angle = -self.steering_input * lock_angle
        turn_speed_per_sec = math.radians(160.0)
        turn_per_timestep = turn_speed_per_sec / hz

        front_left_joint, front_right_joint = self.joints[2:4]
        angle_now = front_left_joint.angle
        angle_to_turn = target_angle - angle_now
        angle_to_turn = np.clip(angle_to_turn, -turn_per_timestep, turn_per_timestep)
        new_angle = angle_now + angle_to_turn
        front_left_joint.SetLimits(new_angle, new_angle)
        front_right_joint.SetLimits(new_angle, new_angle)

        if abs(self.throttle_input) > 1e-3:
            desired_speed = (
                self.throttle_input * self.tires[0].max_forward_speed
                if self.throttle_input > 0
                else self.throttle_input * abs(self.tires[0].max_backward_speed)
            )

            for tire in self.tires:
                current_forward_normal = tire.body.GetWorldVector((0.0, 1.0))
                current_speed = tire.forward_velocity.dot(current_forward_normal)
                # proportional control toward desired speed
                force = (
                    np.clip(desired_speed - current_speed, -1.0, 1.0)
                    * tire.max_drive_force
                )
                tire.body.ApplyForce(
                    tire.current_traction * force * current_forward_normal,
                    tire.body.worldCenter,
                    True,
                )


__all__ = ["Car"]
