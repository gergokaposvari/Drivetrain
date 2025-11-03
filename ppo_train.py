from src.topdown.car_env import CarEnv

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

ENV_ID = "CarGame-v2"

gym.register(
            id=ENV_ID,
            entry_point="src.topdown.car_env:CarEnv",
            max_episode_steps=500,
        )

vec_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs={})  # CarEnv ignores render_mode arg internally


# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=1000000)
# model.save("ppo_topdown_car")


model = PPO.load("ppo_topdown_car")
obs = vec_env.reset()
step = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if step % 200 == 0:
        try:
            car_center = tuple(vec_env.envs[0].simulation.car.body.worldCenter)
        except Exception:
            car_center = None
        print(f"[ppo_train] step={step} action={action} rewards={rewards} car_center={car_center}")
    step += 1



