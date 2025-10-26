from src.topdown.control import DiscreteControl, control_to_keys, enumerate_controls


def test_control_to_keys_mapping() -> None:
    assert control_to_keys(DiscreteControl(throttle=1, steer=-1)) == {"up", "left"}
    assert control_to_keys(DiscreteControl(throttle=-1, steer=1)) == {"down", "right"}
    assert control_to_keys(DiscreteControl(throttle=0, steer=0)) == set()


def test_enumerate_controls_has_expected_combinations() -> None:
    controls = enumerate_controls()
    assert len(controls) == 9
    assert DiscreteControl(throttle=1, steer=1) in controls
    assert DiscreteControl(throttle=-1, steer=-1) in controls
