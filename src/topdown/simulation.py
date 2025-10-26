"""World setup and stepping utilities mirroring the example."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from Box2D import b2Vec2, b2World  # type: ignore import-untyped

from .car import Car
from .contact import TopDownContactListener
from .ground import GroundArea
from .sensors import RaycastConfig, RaycastSample, RaycastSensors
from .timing import LapResult, TimingManager, TimingState
from .track import Track, TrackRecords, TrackSurface, default_track
from .track_persistence import update_fastest_lap


class Simulation:
    """Encapsulates Box2D world management for the top-down car."""

    def __init__(
        self,
        track: Track | None = None,
        *,
        track_file: Path | None = None,
        sensor_config: RaycastConfig | None = None,
    ) -> None:
        self.track = track or default_track()
        self.track_file = track_file
        self.world = b2World(gravity=(0.0, 0.0))
        self._contact_listener = TopDownContactListener(self._handle_tire_contact)
        self.world.contactListener = self._contact_listener

        spawn_angle = self._compute_spawn_angle(self.track)
        self.car = Car(self.world, position=self.track.spawn_point, angle=spawn_angle)
        self._boundary_segments: List[Tuple[b2Vec2, b2Vec2]] = []
        self._surface_drawables: List[Tuple[TrackSurface, List[b2Vec2]]] = []
        self._road_segment_candidates: List[Tuple[Vec2, Vec2]] = []
        self._road_segments: List[Tuple[Vec2, Vec2]] = []
        self._raycast_sensors = RaycastSensors(config=sensor_config)
        self._create_static_geometry()

        records = self.track.records
        best_lap_time = records.fastest_lap_time if isinstance(records, TrackRecords) else None
        best_sector_times = (
            tuple(records.fastest_sector_times)
            if isinstance(records, TrackRecords) and records.fastest_sector_times is not None
            else None
        )
        if (
            self.track.control_points
            and self.track.widths
            and len(self.track.control_points) == len(self.track.widths)
        ):
            if best_sector_times is not None and len(best_sector_times) != len(self.track.control_points):
                best_sector_times = None
            self._timing = TimingManager(
                self.track.control_points,
                self.track.widths,
                self.track.sector_lines,
                lap_callback=self._handle_lap_result,
                best_lap_time=best_lap_time,
                best_sector_times=best_sector_times,
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

    def reset(
        self,
        *,
        position: Tuple[float, float] | None = None,
        angle: float | None = None,
        linear_velocity: Tuple[float, float] | None = None,
        angular_velocity: float | None = None,
    ) -> None:
        """Reset the car pose and clear timers."""
        target_position = position if position is not None else self.track.spawn_point
        target_angle = angle if angle is not None else self._compute_spawn_angle(self.track)
        self.car.reset(target_position, target_angle)
        if linear_velocity is not None:
            self.car.body.linearVelocity = b2Vec2(*linear_velocity)
        if angular_velocity is not None:
            self.car.body.angularVelocity = angular_velocity
        if self._timing is not None:
            self._timing.reset_run()

    def sample_sensors(self) -> RaycastSample:
        """Return the latest raycast sensor measurements."""
        origin_vec = self.car.body.position
        forward_vec = self.car.forward_vector
        origin = (float(origin_vec.x), float(origin_vec.y))
        forward = (float(forward_vec.x), float(forward_vec.y))
        return self._raycast_sensors.sample(origin, forward)

    def car_speed(self) -> float:
        return self.car.forward_speed

    def front_wheel_angle(self) -> float:
        return self.car.front_wheel_angle

    def sensor_angles(self) -> Sequence[float]:
        return self._raycast_sensors.angles

    def sensor_max_distance(self) -> float:
        return self._raycast_sensors.max_distance

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
        self._road_segment_candidates.clear()
        for surface in self.track.surfaces:
            self._create_surface(surface)
        self._finalize_road_segments()

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
            fixture.userData = {"surface": surface.name}
            world_vertices = [transform * b2Vec2(*vertex) for vertex in polygon]
            world_polygons.append(world_vertices)
            if surface.name == "road":
                self._add_road_polygon(world_vertices)
        self._surface_drawables.append((surface, world_polygons))

    def _add_road_polygon(self, vertices: Sequence[b2Vec2]) -> None:
        coords = [(float(vertex.x), float(vertex.y)) for vertex in vertices]
        count = len(coords)
        if count < 2:
            return
        for index in range(count):
            start = coords[index]
            end = coords[(index + 1) % count]
            self._road_segment_candidates.append((start, end))

    def _finalize_road_segments(self) -> None:
        if not self._road_segment_candidates:
            self._road_segments = []
            self._raycast_sensors.set_segments([])
            return

        counts: Dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        segment_map: Dict[tuple[tuple[int, int], tuple[int, int]], Tuple[Vec2, Vec2]] = {}
        for start, end in self._road_segment_candidates:
            key = _edge_key(start, end)
            counts[key] = counts.get(key, 0) + 1
            segment_map.setdefault(key, (start, end))

        self._road_segments = [
            segment_map[key] for key, occurrences in counts.items() if occurrences == 1
        ]
        self._raycast_sensors.set_segments(self._road_segments)

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

    def _handle_lap_result(self, result: LapResult) -> None:
        if self._timing is None or not result.new_best_lap or result.lap_time is None:
            return
        try:
            best_sectors = tuple(self._timing.best_sector_times)
            records = TrackRecords(
                fastest_lap_time=self._timing.best_lap_time,
                fastest_sector_times=best_sectors,
            )
            self.track = replace(self.track, records=records)
            if self.track_file is not None:
                update_fastest_lap(self.track_file, result.lap_time, best_sectors)
        except OSError:
            # Persisting lap data is a best-effort operation.
            pass


def _edge_key(start: Vec2, end: Vec2, scale: float = 1000.0) -> tuple[tuple[int, int], tuple[int, int]]:
    ax = round(start[0] * scale)
    ay = round(start[1] * scale)
    bx = round(end[0] * scale)
    by = round(end[1] * scale)
    first = (ax, ay)
    second = (bx, by)
    return (first, second) if first <= second else (second, first)


__all__ = ["Simulation"]
