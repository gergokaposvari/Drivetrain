"""Training loop entry point with visual and vectorised modes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pygame

from src.topdown.gym_env import TopDownCarEnv
from src.topdown.vector_env import make_vector_env


DEFAULT_TRACK = "tracks/hand_drawn.json"
DEFAULT_HISTORY = 200


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class RandomAgent:
    """Placeholder agent that samples random actions and stores episodes."""

    action_shape: Tuple[int, ...]
    replay_buffer: List[Transition] = field(default_factory=list)
    _action_choices: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],  # accelerate
                [1, 0, 1, 0],  # accelerate + left
                [1, 0, 0, 1],  # accelerate + right
                [0, 1, 0, 0],  # brake
                [0, 0, 1, 0],  # left only
                [0, 0, 0, 1],  # right only
            ],
            dtype=int,
        )
    )

    def act(self, observation: np.ndarray) -> np.ndarray:
        choice = np.random.randint(len(self._action_choices))
        return self._action_choices[choice].astype(int)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)

    def train(self) -> None:
        if self.replay_buffer:
            self.replay_buffer.clear()


@dataclass
class TrainerProgress:
    """Persistent statistics captured between training runs."""

    episodes_completed: int = 0
    best_reward: float | None = None
    reward_history: List[float] = field(default_factory=list)
    history_limit: int = DEFAULT_HISTORY

    def record(self, reward: float) -> None:
        self.episodes_completed += 1
        if self.best_reward is None or reward > self.best_reward:
            self.best_reward = reward
        self.reward_history.append(reward)
        if len(self.reward_history) > self.history_limit:
            excess = len(self.reward_history) - self.history_limit
            del self.reward_history[:excess]

    def to_dict(self) -> dict:
        return {
            "episodes_completed": self.episodes_completed,
            "best_reward": self.best_reward,
            "reward_history": self.reward_history,
            "history_limit": self.history_limit,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TrainerProgress":
        if not payload:
            return cls()
        return cls(
            episodes_completed=int(payload.get("episodes_completed", 0)),
            best_reward=payload.get("best_reward"),
            reward_history=list(payload.get("reward_history", [])),
            history_limit=int(payload.get("history_limit", DEFAULT_HISTORY)),
        )


def load_progress(path: Path | None) -> TrainerProgress:
    if path is None or not path.exists():
        return TrainerProgress()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return TrainerProgress()
    return TrainerProgress.from_dict(data)


def save_progress(path: Path | None, progress: TrainerProgress) -> None:
    if path is None:
        return
    try:
        path.write_text(json.dumps(progress.to_dict(), indent=2) + "\n")
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down car training harness")
    parser.add_argument("--track", default=DEFAULT_TRACK, help="Track file or name to load")
    parser.add_argument(
        "--vector-envs",
        type=int,
        default=1,
        help="Number of simultaneous environments (1 for visual interactive mode)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to run in vector mode")
    parser.add_argument("--asynchronous", action="store_true", help="Use AsyncVectorEnv backend")
    parser.add_argument(
        "--checkpoint",
        default="training_progress.json",
        help="Path to JSON checkpoint storing aggregated training progress",
    )
    return parser.parse_args()


def should_quit() -> bool:
    keys = pygame.key.get_pressed()
    return keys[pygame.K_ESCAPE]


def run_episode(env: TopDownCarEnv, agent: RandomAgent) -> Tuple[float, bool]:
    observation, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.act(observation)
        env.render()
        next_observation, reward, terminated, truncated, _ = env.step(action)

        agent.observe(
            Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=terminated or truncated,
            )
        )

        total_reward += reward
        observation = next_observation
        done = terminated or truncated

        if should_quit():
            return total_reward, True
    return total_reward, False


def run_visual(track: str, checkpoint: Path | None) -> None:
    progress = load_progress(checkpoint)
    env = TopDownCarEnv(render_mode="human", track=track)
    agent = RandomAgent(action_shape=env.action_space.shape)

    episode_index = progress.episodes_completed
    print(
        f"Loaded progress: {progress.episodes_completed} episodes completed, "
        f"best reward {progress.best_reward if progress.best_reward is not None else 'n/a'}."
    )
    try:
        while True:
            total_reward, interrupted = run_episode(env, agent)
            agent.train()
            progress.record(total_reward)
            save_progress(checkpoint, progress)
            best = progress.best_reward if progress.best_reward is not None else float("nan")
            print(
                f"Episode {episode_index:03d} reward: {total_reward:.3f} "
                f"(best {best:.3f}, total {progress.episodes_completed})"
            )
            episode_index += 1
            if interrupted:
                print("Escape pressed; stopping training loop.")
                break
    finally:
        env.close()
        pygame.quit()
        sys.exit(0)


def run_parallel(track: str, num_envs: int, episodes: int, asynchronous: bool, checkpoint: Path | None) -> None:
    progress = load_progress(checkpoint)
    print(
        f"Loaded progress: {progress.episodes_completed} total episodes, "
        f"best reward {progress.best_reward if progress.best_reward is not None else 'n/a'}."
    )
    env = make_vector_env(
        num_envs,
        env_kwargs={"render_mode": None, "track": track},
        asynchronous=asynchronous,
    )
    try:
        total_completed = 0
        reward_sums = np.zeros(num_envs, dtype=np.float64)
        obs, _ = env.reset()
        while total_completed < episodes:
            actions = env.action_space.sample()
            obs, rewards, terminated, truncated, info = env.step(actions)
            reward_sums += rewards
            done = np.logical_or(terminated, truncated)

            if np.any(done):
                for idx, flag in enumerate(done):
                    if flag:
                        total_completed += 1
                        episode_reward = reward_sums[idx]
                        progress.record(float(episode_reward))
                        save_progress(checkpoint, progress)
                        print(
                            f"[Env {idx}] Episode reward: {episode_reward:.3f} "
                            f"(total {progress.episodes_completed}, best {progress.best_reward:.3f})"
                        )
                        reward_sums[idx] = 0.0
                        if total_completed >= episodes:
                            break
                if total_completed >= episodes:
                    break
        print(f"Completed {episodes} episodes across {num_envs} environments.")
    finally:
        env.close()
        sys.exit(0)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if args.vector_envs <= 1:
        run_visual(args.track, checkpoint_path)
    else:
        run_parallel(args.track, args.vector_envs, args.episodes, args.asynchronous, checkpoint_path)


if __name__ == "__main__":
    main()
