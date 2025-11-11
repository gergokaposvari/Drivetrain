from pathlib import Path

from src.topdown.simulation import Simulation
from src.topdown.track_loader import load_track


def test_timing_starts_with_analog_controls_without_keys():
    track = load_track(Path("tracks/simple.json"))
    sim = Simulation(track=track, track_file=Path("tracks/simple.json"))
    state = sim.timing_state
    assert state is not None and not state.running

    sim.car.throttle_input = 0.5
    sim.step(1 / 60.0, 6, 2, keys=None)

    state = sim.timing_state
    assert state is not None and state.running
