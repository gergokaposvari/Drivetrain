from src.topdown.vector_env import make_vector_env


def test_vector_env_batch_step():
    env = make_vector_env(2, env_kwargs={"render_mode": None})
    obs, info = env.reset()
    assert obs.shape[0] == 2
    actions = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    env.close()
