"""World setup and stepping utilities mirroring the example."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from Box2D import b2Vec2, b2World  # type: ignore import-untyped

from .car import Car
from .contact import TopDownContactListener
from .ground import GroundArea
from .timing import TimingManager, TimingState
from .track import Track, TrackSurface, default_track


class Simulation:
    """Encapsulates Box2D world management for the top-down car."""

    def __init__(self, track: Track | None = None) -> None:
        self.track = track or default_track()
        self.world = b2World(gravity=(0.0, 0.0))
        self._contact_listener = TopDownContactListener(self._handle_tire_contact)
        self.world.contactListener = self._contact_listener

        spawn_angle = self._compute_spawn_angle(track)
        self.car = Car(self.world, position=self.track.spawn_point, angle=spawn_angle)
        self._boundary_segments: List[Tuple[b2Vec2, b2Vec2]] = []
        self._surface_drawables: List[Tuple[TrackSurface, List[b2Vec2]]] = []
        self._create_static_geometry()

        if track.control_points and track.widths:
            self._timing = TimingManager(track.control_points, track.widths, track.sector_lines)
        else:
            self._timing = None

        self._tire_contact_names: Dict[object, Set[str]] = {tire: set() for tire in self.car.tires}
        self._tires_on_road = len(self.car.tires)
        self._update_tire_contact_state()

    def step(
        self,
        time_step: float,
        velocity_iterations: int,
        position_iterations: int,
        keys: Set[str],
    ) -> None:
        hz = 1.0 / time_step if time_step > 0 else 60.0
        self.car.update(keys, hz)
        self.world.Step(time_step, velocity_iterations, position_iterations)
        self.world.ClearForces()

        if self._timing is not None:
            self._timing.update(time_step, bool(keys), tuple(self.car.body.position))

    def _create_static_geometry(self) -> None:
        boundary_points = list(self.track.boundary)
        if boundary_points[0] != boundary_points[-1]:
            boundary_points.append(boundary_points[0])
        boundary_body = self.world.CreateStaticBody(position=(0.0, 0.0))
        boundary_body.CreateEdgeChain(boundary_points)
        transform = boundary_body.transform
        self._boundary_segments.clear()
        for start, end in zip(boundary_points, boundary_points[1:]):
            start_vec = transform * b2Vec2(*start)
            end_vec = transform * b2Vec2(*end)
            self._boundary_segments.append((start_vec, end_vec))

        self._surface_drawables.clear()
        for surface in self.track.surfaces:
            self._create_surface(surface)

    @property
    def boundary_segments(self) -> Sequence[Tuple[b2Vec2, b2Vec2]]:
        return tuple(self._boundary_segments)

    @property
    def surfaces(self) -> Sequence[Tuple[TrackSurface, Sequence[Sequence[b2Vec2]]]]:
        return tuple(
            (surface, tuple(tuple(vertex for vertex in polygon) for polygon in polygons))
            for surface, polygons in self._surface_drawables
        )

    def _create_surface(self, surface: TrackSurface) -> None:
        body = self.world.CreateStaticBody(
            userData={"obj": GroundArea(name=surface.name, friction_modifier=surface.friction_modifier)}
        )
        transform = body.transform
        world_polygons: List[List[b2Vec2]] = []
        for polygon in surface.polygons:
            fixture = body.CreatePolygonFixture(vertices=polygon, density=0.0)
            fixture.sensor = True
            world_vertices = [transform * b2Vec2(*vertex) for vertex in polygon]
            world_polygons.append(world_vertices)
        self._surface_drawables.append((surface, world_polygons))

    def _handle_tire_contact(self, tire, area: GroundArea, began: bool) -> None:
        names = self._tire_contact_names.setdefault(tire, set())
        if began:
            names.add(area.name)
        else:
            names.discard(area.name)
        self._update_tire_contact_state()

    def _update_tire_contact_state(self) -> None:
        if self._timing is not None:
            self._timing.set_on_road(True)

    def _compute_spawn_angle(self, track: Track) -> float:
        if track.spawn_direction is None:
            return 0.0
        dx, dy = track.spawn_direction
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return 0.0
        dx /= length
        dy /= length
        import math

        return math.atan2(-dx, dy)

    @property
    def timing_state(self) -> TimingState | None:
        return self._timing.state if self._timing is not None else None

    @property
    def timing_segments(self) -> Sequence[Tuple[Vec2, Vec2]]:
        if self._timing is not None:
            return self._timing.segments
        return ()


__all__ = ["Simulation"]
