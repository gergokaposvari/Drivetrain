from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pygame
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.topdown.rl_agent import DriverAgent

ENV_ID = "CarGame-v2"


def _register_env() -> None:
    if ENV_ID not in gym.registry:
        gym.register(
            id=ENV_ID,
            entry_point="src.topdown.gym_env:TopdownCarEnv",
            max_episode_steps=3000,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualise the car RL agent."
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Q-learning learning rate."
    )
    parser.add_argument(
        "--epsilon-start", type=float, default=1.0, help="Initial exploration rate."
    )
    parser.add_argument(
        "--epsilon-final", type=float, default=0.1, help="Minimum exploration rate."
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        help="Per-episode epsilon decay. Defaults to annealing over half the episodes.",
    )
    parser.add_argument(
        "--qtable",
        type=Path,
        default=Path("artifacts/q_table.pkl"),
        help="Path to save/load the Q-table.",
    )
    parser.add_argument(
        "--load-qtable",
        action="store_true",
        help="Load an existing Q-table before training.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the Q-table after training.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record the final greedy episode to a video file.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("videos"),
        help="Directory where episode videos will be written.",
    )
    parser.add_argument(
        "--video-prefix",
        default="car_rl",
        help="Filename prefix for recorded videos.",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=2000,
        help="Milliseconds to keep the window open after training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_env()

    n_episodes = args.episodes
    learning_rate = args.learning_rate
    epsilon_start = args.epsilon_start
    epsilon_final = args.epsilon_final
    epsilon_decay = (
        args.epsilon_decay
        if args.epsilon_decay is not None
        else epsilon_start / max(1, n_episodes / 2)
    )

    env = gym.make_vec(
        ENV_ID,
        render_mode="human",
        num_envs=4,
        vectorization_mode="sync",
    )

    agent = DriverAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        final_epsilon=epsilon_final,
    )

    if args.load_qtable and args.qtable.exists():
        agent.load(args.qtable)

    obs, info = env.reset()
    env.render()

    episodes_completed = 0
    pbar = tqdm(total=n_episodes, desc="Training episodes")

    while episodes_completed < n_episodes:
        actions = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        env.render()

        done = np.logical_or(terminated, truncated)
        agent.update(obs, actions, reward, done, next_obs)
        obs = next_obs

        if np.any(done):
            remaining = n_episodes - episodes_completed
            finished = int(np.sum(done))
            increment = min(finished, remaining)
            if increment > 0:
                for _ in range(increment):
                    agent.decay_epsilon()
                episodes_completed += increment
                pbar.update(increment)
    pbar.close()

    if not args.no_save:
        agent.save(args.qtable)

    pygame.time.wait(args.wait_ms)
    env.close()

    if args.record_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        eval_env = gym.make(ENV_ID, render_mode="rgb_array")
        video_env = RecordVideo(
            eval_env,
            video_folder=str(args.video_dir),
            episode_trigger=lambda episode: episode == 0,
            name_prefix=args.video_prefix,
        )
        agent.set_env(video_env)
        epsilon_backup = agent.epsilon
        agent.epsilon = 0.0

        obs, info = video_env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated

        video_env.close()
        agent.epsilon = epsilon_backup


if __name__ == "__main__":
    main()
