"""World setup and stepping utilities mirroring the example."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Set, Tuple

from Box2D import b2Vec2, b2World  # type: ignore import-untyped

from .car import Car
from .contact import TopDownContactListener
from .ground import GroundArea
from .timing import TimingManager, TimingState
from .track import Track, TrackSurface, Vec2, default_track
from .state import SimulationSnapshot, vec2_to_tuple


class Simulation:
    """Encapsulates Box2D world management for the top-down car."""

    def __init__(
        self,
        track: Track | None = None,
        *,
        on_best_lap: Callable[[float, Sequence[float]], None] | None = None,
    ) -> None:
        self.track = track or default_track()
        self.world = b2World(gravity=(0.0, 0.0))
        self._contact_listener = TopDownContactListener(self._handle_tire_contact)
        self.world.contactListener = self._contact_listener
        self._sensor_angles_rad = [math.radians(angle) for angle in (-60, -30, -15, 0, 15, 30, 60)]
        self._sensor_range = 200.0
        self._road_polygons: List[Sequence[Vec2]] = []
        self._surface_sensor_polygons: List[tuple[TrackSurface, Sequence[Vec2]]] = []

        spawn_angle = self._compute_spawn_angle(self.track)
        self.car = Car(self.world, position=self.track.spawn_point, angle=spawn_angle)
        self._boundary_segments: List[Tuple[b2Vec2, b2Vec2]] = []
        self._surface_drawables: List[Tuple[TrackSurface, List[b2Vec2]]] = []
        self._create_static_geometry()

        if self.track.control_points and self.track.widths:
            self._timing = TimingManager(
                self.track.control_points,
                self.track.widths,
                self.track.sector_lines,
                best_lap_time=self.track.best_lap_time,
                best_sector_times=self.track.best_lap_sector_times,
                on_best_lap=on_best_lap,
            )
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
        self._surface_sensor_polygons.clear()
        self._road_polygons.clear()
        for surface in self.track.surfaces:
            self._create_surface(surface)
        if not self._road_polygons and self._surface_sensor_polygons:
            best_surface = max(self._surface_sensor_polygons, key=lambda item: item[0].friction_modifier)[0]
            self._road_polygons = [
                polygon for surface, polygon in self._surface_sensor_polygons if surface is best_surface
            ]

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
        sensor_polygons: List[Sequence[Vec2]] = []
        for polygon in surface.polygons:
            fixture = body.CreatePolygonFixture(vertices=polygon, density=0.0)
            fixture.sensor = True
            world_vertices = [transform * b2Vec2(*vertex) for vertex in polygon]
            world_polygons.append(world_vertices)
            sensor_polygons.append(tuple(vec2_to_tuple(vertex) for vertex in world_vertices))
        self._surface_drawables.append((surface, world_polygons))
        self._surface_sensor_polygons.extend((surface, polygon) for polygon in sensor_polygons)
        if surface.name.lower() == "road":
            self._road_polygons.extend(sensor_polygons)

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

    @property
    def tire_contact_names(self) -> Sequence[Sequence[str]]:
        return tuple(
            tuple(sorted(self._tire_contact_names.get(tire, ())))
            for tire in self.car.tires
        )

    @property
    def sensor_distances(self) -> Sequence[float]:
        return self._compute_sensor_distances()

    @property
    def sensor_range(self) -> float:
        return self._sensor_range

    def snapshot(self, *, elapsed_time: float = 0.0, step_index: int = 0) -> SimulationSnapshot:
        """Return an immutable view of the current simulation state."""
        return SimulationSnapshot(
            car=self.car.snapshot(self._tire_contact_names),
            timing=self.timing_state,
            tire_contacts=self.tire_contact_names,
            sensor_distances=self.sensor_distances,
            elapsed_time=elapsed_time,
            step_index=step_index,
        )

    def _compute_sensor_distances(self) -> Sequence[float]:
        if not self._road_polygons:
            return tuple(self._sensor_range for _ in self._sensor_angles_rad)

        origin_vec = self.car.body.position
        origin = (float(origin_vec.x), float(origin_vec.y))
        forward = self.car.body.GetWorldVector((0.0, 1.0))
        right = self.car.body.GetWorldVector((1.0, 0.0))
        forward_tuple = (float(forward.x), float(forward.y))
        right_tuple = (float(right.x), float(right.y))

        distances: List[float] = []
        for angle in self._sensor_angles_rad:
            dir_x = forward_tuple[0] * math.cos(angle) + right_tuple[0] * math.sin(angle)
            dir_y = forward_tuple[1] * math.cos(angle) + right_tuple[1] * math.sin(angle)
            length = math.hypot(dir_x, dir_y)
            if length == 0:
                distances.append(self._sensor_range)
                continue
            direction = (dir_x / length, dir_y / length)
            distance = self._sensor_range
            for polygon in self._road_polygons:
                for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                    hit = self._ray_segment_intersection(origin, direction, start, end)
                    if hit is not None and hit < distance:
                        distance = hit
            distances.append(distance)
        return tuple(distances)

    @staticmethod
    def _ray_segment_intersection(
        origin: Vec2,
        direction: Vec2,
        start: Vec2,
        end: Vec2,
    ) -> float | None:
        sx, sy = start
        ex, ey = end
        ox, oy = origin
        dx, dy = direction
        seg_dx = ex - sx
        seg_dy = ey - sy

        denom = dx * seg_dy - dy * seg_dx
        if abs(denom) < 1e-8:
            return None

        diff_x = sx - ox
        diff_y = sy - oy

        t = (diff_x * seg_dy - diff_y * seg_dx) / denom
        u = (diff_x * dy - diff_y * dx) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None


__all__ = ["Simulation"]
