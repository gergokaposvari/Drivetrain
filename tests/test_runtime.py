from src.topdown.runtime import SimulationConfig, SimulationSession
from src.topdown.track import default_track


def test_session_step_advances_time() -> None:
    track = default_track()
    config = SimulationConfig(time_step=0.05)
    session = SimulationSession(track=track, config=config)

    initial = session.snapshot()
    assert initial.step_index == 0
    assert initial.elapsed_time == 0.0

    after_step = session.step([])
    assert after_step.step_index == 1
    assert after_step.elapsed_time == config.time_step
    assert after_step.car.tire_states
    assert len(after_step.sensor_distances) == 7
