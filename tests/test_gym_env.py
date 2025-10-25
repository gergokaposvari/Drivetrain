import numpy as np

from src.topdown.gym_env import TopDownCarEnv


def test_env_reset_and_step():
    env = TopDownCarEnv()
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (14,)
    assert "tire_contacts" in info

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (14,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "timing_state" in step_info
    assert "sensor_distances" in step_info
    env.close()
