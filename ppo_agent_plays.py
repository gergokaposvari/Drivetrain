import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from src.topdown.car_env import CarEnv

ENV_ID = "CarGame-v2"

gym.register(
    id=ENV_ID,
    entry_point="src.topdown.car_env:CarEnv",
    max_episode_steps=3500,
)

vec_env = make_vec_env(
    ENV_ID,
    n_envs=1,
    env_kwargs={"render_mode": "human"},
)


model = SAC.load("models/dist_to_sector_sb3_sac")
obs = vec_env.reset()

print("Starting simulation... Waiting for first lap completion.")

# Data storage
telemetry_data = {
    "speed": [],
    "wheel_angle": [],
    "throttle": [],
    # "brake": [] # Brake removed
}

# We need to detect when the lap time changes/is set.
# Initially, it should be None.
initial_lap_time_checked = False
last_lap_time = None

step = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = vec_env.step(action)
    vec_env.render("human")

    # If crash/reset occurs, clear data
    if dones[0]:
        print("Episode ended (crash or reset). Restarting data collection for new lap.")
        telemetry_data = {
            "speed": [],
            "wheel_angle": [],
            "throttle": [],
            # "brake": []
        }
        # Update last_lap_time to whatever the new episode starts with (likely None)
        # But we can't easily get info after reset from here because vec_env auto-resets and returns obs.
        # However, the next step's info will contain the state.
        # We can assume we reset logic.
        initial_lap_time_checked = False
        last_lap_time = None

    if step % 200 == 0:
        print(f"Step {step}: Driving...")

    step += 1

print(f"Collected {len(telemetry_data['speed'])} steps of data. Generating plots...")


# Smoothing function
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


# Apply smoothing to wheel angle
# Use a window size of ~50 steps for stronger smoothing (approx 5/6 sec at 60Hz)
smoothed_wheel_angle = smooth(telemetry_data["wheel_angle"], 50)


# Plotting
# Changed to 4 rows
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

steps_range = np.arange(len(telemetry_data["speed"]))

# Speed
axs[0].plot(steps_range, telemetry_data["speed"], label="Speed", color="blue")
axs[0].set_ylabel("Speed")
axs[0].legend(loc="upper right")
axs[0].grid(True)
axs[0].set_title("Vehicle Telemetry Over First Completed Lap")

# Raw Wheel Angle
axs[1].plot(
    steps_range,
    telemetry_data["wheel_angle"],
    label="Raw Wheel Angle",
    color="orange",
    alpha=0.7,
)
axs[1].axhline(0, color="black", linestyle="--", linewidth=1)
axs[1].set_ylabel("Angle (rad)")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# Smoothed Wheel Angle
axs[2].plot(
    steps_range, smoothed_wheel_angle, label="Smoothed Wheel Angle", color="darkorange"
)
axs[2].axhline(0, color="black", linestyle="--", linewidth=1)
axs[2].set_ylabel("Angle (rad)")
axs[2].legend(loc="upper right")
axs[2].grid(True)

# Throttle
axs[3].plot(steps_range, telemetry_data["throttle"], label="Throttle", color="green")
axs[3].set_ylabel("Throttle")
axs[3].set_xlabel("Step")
axs[3].legend(loc="upper right")
axs[3].grid(True)
plt.tight_layout()
print("Saving plot to docs/lap_telemetry.png...")
plt.savefig("docs/lap_telemetry.png")
