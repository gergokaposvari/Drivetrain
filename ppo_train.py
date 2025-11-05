from src.topdown.car_env import CarEnv

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

ENV_ID = "CarGame-v2"

gym.register(
    id=ENV_ID,
    entry_point="src.topdown.car_env:CarEnv",
    max_episode_steps=3500,
)

vec_env = make_vec_env(
    ENV_ID,
    n_envs=4,
    env_kwargs={"render_mode": None},
)


model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=7000000, progress_bar=True)
# model.save("ppo_topdown_car")
#
# model = PPO.load("ppo_topdown_car_new_reward", vec_env)
model.learn(total_timesteps=15000000, reset_num_timesteps=False, progress_bar=True)
model.save("charles_leclerc")
