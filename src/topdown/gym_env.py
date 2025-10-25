"""Gymnasium-ready environment wrapping the top-down car simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Set

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

from .runtime import SimulationConfig, SimulationSession
from .state import SimulationSnapshot
from .track import Track, default_track
from .track_loader import load_track


@dataclass(frozen=True)
class RewardWeights:
    """Simple shaping coefficients used by the demo reward."""

    sector_completion: float = 1.0
    sector_improvement: float = 1.0
    best_lap_bonus: float = 5.0
    forward_speed: float = 0.01


class TopDownCarEnv(gym.Env[np.ndarray, np.ndarray]):
    """Demonstration Gymnasium environment for reinforcement learning."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        *,
        track: Track | str | Path | None = None,
        render_mode: str | None = None,
        config: SimulationConfig | None = None,
        max_steps: int = 1800,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode '{render_mode}'")

        self._weights = reward_weights or RewardWeights()
        self._max_steps = max(1, int(max_steps))
        self._config = config or SimulationConfig()

        if track is None:
            self._base_track = default_track()
        elif isinstance(track, Track):
            self._base_track = track
        else:
            self._base_track = load_track(Path(track))

        self._session = SimulationSession(track=self._base_track, config=self._config)
        self._step_count = 0
        self._np_random = None

        sensor_range = self._session.simulation.sensor_range
        self.observation_space = self._make_observation_space(sensor_range)
        self.action_space = spaces.MultiBinary(4)

        self._sensor_range = sensor_range
        self._sector_completion_tracker: set[int] = set()
        self._sector_improvement_tracker: set[int] = set()
        self._last_best_lap_time: float | None = None
        self._last_timing_time: float | None = None

        self._renderer = None
        self._screen = None
        self._clock = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._np_random, _ = seeding.np_random(seed)
        track_override = options.get("track") if options else None
        if isinstance(track_override, (str, Path)):
            active_track = load_track(Path(track_override))
        elif isinstance(track_override, Track):
            active_track = track_override
        else:
            active_track = None

        snapshot = self._session.reset(track=active_track)
        self._sensor_range = self._session.simulation.sensor_range
        self.observation_space = self._make_observation_space(self._sensor_range)
        self._step_count = 0
        self._sector_completion_tracker.clear()
        self._sector_improvement_tracker.clear()
        timing = snapshot.timing
        self._last_best_lap_time = timing.best_lap_time if timing is not None else None
        self._last_timing_time = timing.current_time if timing is not None else None
        observation = self._build_observation(snapshot)
        info = self._build_info(snapshot)
        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        controls = self._multi_binary_to_controls(action)
        snapshot = self._session.step(controls)
        self._step_count += 1

        reward = self._compute_reward(snapshot)
        observation = self._build_observation(snapshot)
        terminated = self._should_terminate(snapshot)
        truncated = self._step_count >= self._max_steps
        info = self._build_info(snapshot)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            raise NotImplementedError("Only human rendering is supported in this demo environment.")

        import pygame  # Imported lazily to keep headless usage lightweight

        if self._renderer is None:
            from .render import RenderConfig, Renderer

            pygame.init()
            config = RenderConfig()
            self._screen = pygame.display.set_mode(config.screen_size)
            pygame.display.set_caption("TopDownCarEnv")
            self._renderer = Renderer(self._screen, config)
            self._clock = pygame.time.Clock()

        pygame.event.pump()
        self._renderer.draw(self._session.simulation)
        assert self._clock is not None
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._renderer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._renderer = None
            self._screen = None
            self._clock = None
        super().close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, snapshot: SimulationSnapshot) -> np.ndarray:
        position = np.array(snapshot.car.position, dtype=np.float32)
        velocity = np.array(snapshot.car.linear_velocity, dtype=np.float32)
        angle = float(snapshot.car.angle)
        sensors = np.array(snapshot.sensor_distances, dtype=np.float32)
        observation = np.concatenate(
            (
                np.array(
                    [
                        position[0],
                        position[1],
                        velocity[0],
                        velocity[1],
                        math.sin(angle),
                        math.cos(angle),
                        float(snapshot.car.angular_velocity),
                    ],
                    dtype=np.float32,
                ),
                sensors,
            )
        )
        return observation

    def _compute_reward(self, snapshot: SimulationSnapshot) -> float:
        timing = snapshot.timing
        if timing is None:
            self._last_timing_time = None
            self._last_best_lap_time = None
            return 0.0

        reward = 0.0
        reset_trackers = False
        if self._last_timing_time is not None and timing.current_time < self._last_timing_time:
            reset_trackers = True

        if reset_trackers:
            self._sector_completion_tracker.clear()
            self._sector_improvement_tracker.clear()

        for index, status in enumerate(timing.sector_statuses):
            if status.completed and index not in self._sector_completion_tracker:
                reward += self._weights.sector_completion
                self._sector_completion_tracker.add(index)
            if status.faster and index not in self._sector_improvement_tracker:
                reward += self._weights.sector_improvement
                self._sector_improvement_tracker.add(index)

        if timing.best_lap_time is not None:
            if (
                self._last_best_lap_time is None
                or timing.best_lap_time < self._last_best_lap_time - 1e-6
            ):
                reward += self._weights.best_lap_bonus
            self._last_best_lap_time = timing.best_lap_time

        velocity = np.array(snapshot.car.linear_velocity, dtype=np.float32)
        angle = float(snapshot.car.angle)
        forward_vec = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)
        forward_speed = float(np.dot(velocity, forward_vec))
        if forward_speed > 0.0:
            reward += self._weights.forward_speed * forward_speed

        self._last_timing_time = timing.current_time
        return reward

    def _should_terminate(self, snapshot: SimulationSnapshot) -> bool:
        timing = snapshot.timing
        if timing is None:
            return False
        return timing.last_lap_time is not None and not timing.invalid

    def _build_info(self, snapshot: SimulationSnapshot) -> dict:
        return {
            "timing_state": snapshot.timing,
            "tire_contacts": snapshot.tire_contacts,
            "step_index": snapshot.step_index,
            "sensor_distances": snapshot.sensor_distances,
        }

    def _multi_binary_to_controls(self, action: Iterable[int]) -> Set[str]:
        bits = np.asarray(action, dtype=int).flatten()
        if bits.shape[0] != 4:
            raise ValueError("Action must contain four elements [up, down, left, right]")
        mapping = ("up", "down", "left", "right")
        return {mapping[idx] for idx, value in enumerate(bits) if value >= 1}

    @staticmethod
    def _make_observation_space(sensor_range: float) -> spaces.Box:
        low = np.concatenate(
            (
                np.full(7, -np.inf, dtype=np.float32),
                np.zeros(7, dtype=np.float32),
            )
        )
        high = np.concatenate(
            (
                np.full(7, np.inf, dtype=np.float32),
                np.full(7, sensor_range, dtype=np.float32),
            )
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)


__all__ = ["TopDownCarEnv", "RewardWeights"]
