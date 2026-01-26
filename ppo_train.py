import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from src.topdown.car_env import CarEnv

ENV_ID = "CarGame-v2"

gym.register(
    id=ENV_ID,
    entry_point="src.topdown.car_env:CarEnv",
    max_episode_steps=3000,
)

# Prefer ROCm/AMD GPU if available; fall back to CPU otherwise.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

vec_env = make_vec_env(
    ENV_ID,
    n_envs=4,
    env_kwargs={"render_mode": None},
)

# Switch to PPO if desired: PPO("MlpPolicy", vec_env, device=DEVICE, verbose=1)
model = SAC("MlpPolicy", vec_env, device=DEVICE, verbose=1)

# model = PPO.load("charles_leclerc", vec_env)


model.learn(total_timesteps=1000000, progress_bar=True)
model.save("dist_to_sector_sb3_sac")
