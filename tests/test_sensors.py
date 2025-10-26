import pytest

from src.topdown.simulation import Simulation
from src.topdown.track import Track, TrackRecords, TrackSurface


def _make_rect_track() -> Track:
    boundary = (
        (-30.0, -30.0),
        (-30.0, 30.0),
        (30.0, 30.0),
        (30.0, -30.0),
    )
    road_polygon = (
        (-10.0, -15.0),
        (10.0, -15.0),
        (10.0, 15.0),
        (-10.0, 15.0),
    )
    grass = TrackSurface(name="grass", polygons=(boundary,), friction_modifier=0.5)
    road = TrackSurface(name="road", polygons=(road_polygon,), friction_modifier=1.0)
    return Track(
        boundary=boundary,
        surfaces=(grass, road),
        spawn_point=(0.0, 0.0),
        spawn_direction=(0.0, 1.0),
        records=TrackRecords(),
    )


def test_raycast_distances_centered_car() -> None:
    simulation = Simulation(track=_make_rect_track())
    sample = simulation.sample_sensors()

    assert len(sample.distances) == 7
    assert len(sample.hits) == 7
    assert all(sample.hits)
    assert list(simulation.sensor_angles()) == [-60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0]
    assert simulation.sensor_max_distance() == pytest.approx(150.0)

    forward_distance = sample.distances[3]  # angle 0
    assert forward_distance == pytest.approx(15.0, rel=1e-3)

    left_distance = sample.distances[0]
    right_distance = sample.distances[-1]
    assert left_distance == pytest.approx(right_distance, rel=1e-3)


def test_reset_clears_velocity_and_wheel_angle() -> None:
    simulation = Simulation(track=_make_rect_track())
    simulation.step(1.0 / 60.0, 6, 2, {"up"})
    assert simulation.car.body.linearVelocity.length > 0

    simulation.reset()
    assert simulation.car.body.linearVelocity.length == pytest.approx(0.0, abs=1e-3)
    assert simulation.car.body.angularVelocity == pytest.approx(0.0, abs=1e-6)
    assert simulation.front_wheel_angle() == pytest.approx(0.0, abs=1e-6)
