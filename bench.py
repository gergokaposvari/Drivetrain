import time
import numpy as np
from src.topdown.vector_env import make_vector_env


def benchmark_vector_env(count: int, steps: int = 1000) -> float:
    env = make_vector_env(count, env_kwargs={"render_mode": None})
    obs, info = env.reset()
    total_actions = np.zeros((count, env.single_action_space.shape[0]), dtype=np.int32)
    start = time.perf_counter()
    for _ in range(steps):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_actions += actions
    elapsed = time.perf_counter() - start
    env.close()
    return steps / elapsed


for count in range(1, 20):
    steps_per_second = benchmark_vector_env(count, steps=1200)
    hz_per_env = steps_per_second / count
    print(
        f"{count:2d} envs → {steps_per_second:8.1f} steps/sec "
        f"({hz_per_env:6.1f} Hz per env)"
    )
